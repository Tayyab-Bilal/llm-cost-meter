"""Build-time guard: fail CI if code can bypass the single choke point.

Usage: python -m llm_cost_meter.guards.ast_guard <path> [<path> ...]

This is a deny-list of known provider clients, not a proof. It does not see: getattr with a variable name,
`vars()`/`globals()` lookups, vendors not listed below, or an alias of `charge_units` (`f = meter.charge_units`).
A wrapper that mimics a third-party interface must subclass the real base class.
Suppress a reviewed line with a `# billing-guard: ok` comment.
"""
import ast
import sys
from collections.abc import Iterator
from pathlib import Path

from ..buckets import SERVICE_BUCKETS

PROVIDER_CLIENTS = {
    "ChatOpenAI", "ChatAnthropic", "OpenAI", "Anthropic", "AsyncOpenAI", "AsyncAnthropic",
    "AzureChatOpenAI", "AzureOpenAI", "AsyncAzureOpenAI", "init_chat_model",  # init_chat_model builds any provider
}
PROVIDER_MODULES = {"openai", "anthropic", "langchain_openai", "langchain_anthropic", "langchain"}
LABEL_CALLS = {"metered_chat_model", "charge_units"}
PACKAGE = "llm_cost_meter"
ALLOWED_FACTORY_MODULES = {f"{PACKAGE}/providers.py"}  # the one place real clients may be built
SUPPRESS = "billing-guard: ok"
# Methods a class only has when it stands in for a third-party interface (chat model, embeddings client).
WRAPPER_METHODS = {
    "invoke", "ainvoke", "stream", "astream", "_generate", "_agenerate", "_astream",
    "embed_documents", "embed_query", "aembed_documents", "aembed_query", "__getattr__",
}


def _module_path(path: Path) -> str:
    """Path from the top-level package dir (found via __init__.py), so a stray providers.py is not exempt."""
    parts, d = [path.name], path.resolve().parent
    while (d / "__init__.py").exists():
        parts.insert(0, d.name)
        d = d.parent
    return "/".join(parts)


def _ref(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _root(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _walk(node: ast.AST, params: frozenset[str]) -> Iterator[tuple[ast.AST, frozenset[str]]]:
    """Walk, carrying the parameter names of the enclosing functions (to allow label forwarding)."""
    yield node, params
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        a = node.args
        params = params | {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if x}
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, params)


def _ignored_ids(tree: ast.AST) -> set[int]:
    """Annotations and isinstance() checks mention a class without constructing it."""
    skip: set[ast.AST] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.arg | ast.AnnAssign) and getattr(n, "annotation", None):
            skip.add(n.annotation)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.returns:
            skip.add(n.returns)
        if isinstance(n, ast.Call) and _ref(n.func) in {"isinstance", "issubclass"} and len(n.args) > 1:
            skip.add(n.args[1])
    return {id(d) for s in skip for d in ast.walk(s)}


def check_file(path: Path) -> list[str]:
    try:
        text = path.read_text()
        tree = ast.parse(text, str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        line = getattr(exc, "lineno", 0) or 0
        return [f"{path}:{line}: cannot parse ({type(exc).__name__}); fix the file so it can be checked"]
    lines = text.splitlines()
    alias: dict[str, str] = {}  # local name -> real name, for any import
    clients: dict[str, str] = {}  # local name -> provider client imported from a provider module
    modules: set[str] = set()  # local names of imported provider modules
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                alias[a.asname or a.name] = a.name
                if (n.module or "").split(".")[0] in PROVIDER_MODULES and a.name in PROVIDER_CLIENTS:
                    clients[a.asname or a.name] = a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in PROVIDER_MODULES:
                    modules.add(a.asname or a.name.split(".")[0])
    allowed = _module_path(path) in ALLOWED_FACTORY_MODULES
    ignored = _ignored_ids(tree)
    problems: list[str] = []

    def flag(node: ast.AST, msg: str) -> None:
        if SUPPRESS not in lines[node.lineno - 1]:
            problems.append(f"{path}:{node.lineno}: {msg}")

    for node, params in _walk(tree, frozenset()):
        if id(node) in ignored:
            continue
        # Any load counts, not just a call: functools.partial(ChatOpenAI) builds one later.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in clients and not allowed:
            flag(node, f"reference to provider client {clients[node.id]}; use metered_chat_model()")
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in PROVIDER_CLIENTS
                and _root(node) in modules and not allowed):
            flag(node, f"reference to provider client {node.attr}; use metered_chat_model()")
        if isinstance(node, ast.ClassDef) and all(_ref(b) == "object" for b in node.bases):
            defined = {n.name for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
            mimics = sorted(WRAPPER_METHODS & defined)
            if mimics:  # SDKs check isinstance(x, BaseChatModel / Embeddings); a look-alike fails that check
                flag(node, f"class {node.name} mimics a third-party interface ({', '.join(mimics)}) without "
                           "subclassing its base class; isinstance checks will fail, so subclass the real base")
        if not isinstance(node, ast.Call):
            continue
        name = alias.get(_ref(node.func), _ref(node.func))
        if name == "getattr" and len(node.args) >= 2 and not allowed:
            lit = node.args[1]
            if isinstance(lit, ast.Constant) and lit.value in PROVIDER_CLIENTS:
                flag(node, f"getattr lookup of provider client {lit.value}")
        if name not in LABEL_CALLS:
            continue
        label = next((k.value for k in node.keywords if k.arg == "service"), None)
        if label is None and name == "metered_chat_model" and node.args:
            label = node.args[0]
        if label is None and any(k.arg is None for k in node.keywords):
            flag(node, "service label hidden in **kwargs; label must be a literal so CI can check it")
        elif isinstance(label, ast.Name) and label.id in params:
            continue  # a wrapper forwarding its own parameter; its callers are checked where they pass a literal
        elif label is not None and not (isinstance(label, ast.Constant) and isinstance(label.value, str)):
            flag(node, "service label must be a string literal so CI can check it")
        elif label is not None and label.value not in SERVICE_BUCKETS:
            flag(node, f"service label {label.value!r} is not in SERVICE_BUCKETS")
    return problems


def main(argv: list[str]) -> int:
    files = [f for a in argv for f in ([Path(a)] if Path(a).is_file() else sorted(Path(a).rglob("*.py")))]
    problems = [p for f in files for p in check_file(f)]
    print("\n".join(problems) if problems else f"ast_guard: {len(files)} files clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
