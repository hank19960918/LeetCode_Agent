#!/usr/bin/env python3

import os, re, json, textwrap, argparse, subprocess, sys, uuid, tempfile
from typing import Any, Dict, List, TypedDict
from datetime import datetime
import os, uuid

# LangChain / LangGraph
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Optional Tavily
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    HAS_TAVILY = True
except Exception:
    HAS_TAVILY = False


class WFState(TypedDict):
    description: str
    examples: List[Dict[str, str]] 
    constraints: str
    follow_up: str
    interface: str
    language: str
    spec_path: str
    workdir: str
    search_needed: bool
    search_context: str
    code: str
    tests: List[Dict[str, Any]]   
    test_result: Dict[str, Any]  
    complexity: Dict[str, str]  
    report_path: str
    augment_tests: bool
    augment_count: int


#### post processing for the generated code
def strip_fences(s: str) -> str:
    """Remove ```python ... ``` fences from LLM output."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

#### get c++ interface method information, ensure generated code matches the interface
def parse_cpp_method_name(interface: str, default: str = "twoSum") -> str:
    if not interface:
        return default

    m_cls = re.search(r'class\s+Solution\s*\{(.*)\};', interface, re.DOTALL)
    if not m_cls:
        return default
    body = m_cls.group(1)

    body = re.sub(r'\b(public|private|protected)\s*:\s*', ' ', body)
    m_fn = re.search(
        r'''(?x)
        ^\s*
        (?:template\s*<[^>]+>\s*)?            
        (?:inline\s+|static\s+|virtual\s+|constexpr\s+|friend\s+)*  
        [A-Za-z_:\<\>\s\*&]+?                 
        \b([A-Za-z_]\w*)\s*                   
        \([^;{}]*\)\s*                        
        (?:;|\{)                              
        ''',
        body,
        re.MULTILINE,
    )
    return m_fn.group(1) if m_fn else default

#### parse problem information from input_file
def parse_flexible_spec_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    t = txt.replace("\r\n", "\n")


    m_iface_cpp = re.search(r'(class\s+Solution\s*\{.*?\};)', t, re.DOTALL)
    m_iface_py  = re.search(r'(class\s+Solution\s*:\s*.*?)(?:\Z|\n{2,}|\nclass |\ndef )', t, re.DOTALL)
    interface = ""
    if m_iface_cpp:
        interface = m_iface_cpp.group(1).strip()
    elif m_iface_py:
        interface = m_iface_py.group(1).strip()

    m_fu = re.search(r'Follow-?up:\s*(.+)', t)
    follow_up = m_fu.group(1).strip() if m_fu else ""

    m_cons = re.search(r'Constraints:\s*(.*?)(?:\n\s*\n|Follow-?up:|class\s+Solution|\Z)', t, re.DOTALL)
    constraints = m_cons.group(1).strip() if m_cons else ""

    ex_blocks = []
    for m in re.finditer(r'Example\s+\d+:\s*(.*?)(?=\nExample\s+\d+:|\Z)', t, re.DOTALL):
        block = m.group(1).strip()
        inp = re.search(r'Input:\s*(.+)', block)
        out = re.search(r'Output:\s*(.+)', block)
        expl = re.search(r'Explanation:\s*(.+)', block)
        ex_blocks.append({
            "input": (inp.group(1).strip() if inp else ""),
            "output": (out.group(1).strip() if out else ""),
            "explanation": (expl.group(1).strip() if expl else "")
        })

    ex1 = re.search(r'\bExample\s+1:', t)
    if ex1:
        desc = t[:ex1.start()].strip()
    else:
        cut = min([p for p in [
            t.find("Constraints:"),
            re.search(r'Follow-?up:', t).start() if re.search(r'Follow-?up:', t) else -1,
            t.find("class Solution")
        ] if p != -1] or [len(t)])
        desc = t[:cut].strip()

    desc = re.sub(r'^(Solved|Topics|premium lock icon|Companies|Hint)\s*', '', desc,
                  flags=re.I|re.M).strip()
    print("[finished] parsing spec file")

    return {
        "description": desc,
        "examples": ex_blocks,
        "constraints": constraints,
        "follow_up": follow_up,
        "interface": interface
    }

#### generated progeram language autodetect from the interface infromation
def autodetect_language_from_interface(interface: str, default: str = "python") -> str:
    if "class Solution {" in interface:
        return "cpp"
    if re.search(r"class\s+Solution\s*:", interface):
        return "python"
    return default


if HAS_TAVILY:
    tavily_tool = TavilySearchResults(max_results=5)
else:
    tavily_tool = None


#### let LLM decide if search is needed, and generate the query
@tool
def tavily_search(q: str) -> str:
    """Use Tavily to search relevant algorithm references or patterns."""
    if not HAS_TAVILY:
        print("[ERROR] Tavily not available")
        return "Tavily not available."
    try:
        res = tavily_tool.invoke({"query": q})
        print("[finished] tavily search")
        return json.dumps(res, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[ERROR] Tavily not available")
        return f"Search error: {e}"


#### LLM initialization
def get_llm():
    return init_chat_model("openai:gpt-4o")



#### Workflow nodes definition
def load_spec_node(state: WFState) -> WFState:
    parsed = parse_flexible_spec_file(state["spec_path"])
    language = state["language"] or autodetect_language_from_interface(parsed.get("interface", ""))
    return {
        "description": parsed["description"],
        "examples": parsed["examples"],
        "constraints": parsed["constraints"],
        "follow_up": parsed["follow_up"],
        "interface": parsed["interface"],
        "language": language,
    }

#### decide using tavily search or not
def decide_search_node(state: WFState) -> WFState:
    llm = get_llm()
    sys_prompt = """You are a senior algorithm engineer.
Given a LeetCode-style problem, decide if internet search could significantly help code generation
(e.g., rare data structures, tricky proofs). Output JSON: {"need_search": true/false, "query": "...", "reason": "..."}."""
    content = f"""Problem:
{state['description']}

Interface:
{state['interface']}

Constraints:
{state['constraints']}

Follow-up:
{state['follow_up']}
"""
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=content)])
    try:
        data = json.loads(str(resp.content))
    except Exception:
        data = {"need_search": False, "query": "", "reason": "fallback-no-json"}

    search_context = ""
    if data.get("need_search") and tavily_tool:
        result = tavily_search.invoke({"q": data.get("query", "")})
        search_context = str(result)
    return {"search_needed": bool(data.get("need_search")), "search_context": search_context}

GEN_CODE_SYS = """You write correct, clean, and efficient code for LeetCode problems.
- Follow the provided function/class interface exactly.
- Prefer optimal time/space complexity.
- Output ONLY the full code (no prose).
"""

#### base on the problem description, interface, constraints, examples, and search context (if any), generate the solution code
def gen_code_node(state: WFState) -> WFState:
    llm = get_llm()
    msgs = [
        SystemMessage(content=GEN_CODE_SYS),
        HumanMessage(content=textwrap.dedent(f"""
        Problem:
        {state['description']}

        Interface (language={state['language']}):
        {state['interface']}

        Constraints:
        {state['constraints']}

        Examples (raw text):
        {json.dumps(state['examples'], ensure_ascii=False, indent=2)}

        Helpful context (if any):
        {state['search_context']}
        """))
    ]
    resp = llm.invoke(msgs)
    code = str(resp.content).strip()
    code = strip_fences(code)
    return {"code": code}

#### extract test cases from the Examples using LLM
def examples_to_tests_via_llm(
    examples: List[Dict[str, str]],
    interface: str,
    language: str,
    solution_code: str = ""
) -> List[Dict[str, Any]]:
    llm = get_llm()

    examples_text = "\n\n".join(
        [f"Example {i+1}:\nInput: {ex.get('input','')}\nOutput: {ex.get('output','')}\nExplanation: {ex.get('explanation','')}"
         for i, ex in enumerate(examples)]
    )

    if solution_code:
        head = solution_code[:6000]
        tail = solution_code[-6000:] if len(solution_code) > 12000 else ""
        solution_excerpt = head + ("\n...\n" if tail else "") + tail
    else:
        solution_excerpt = ""

    sys_prompt = """You extract structured test cases from LeetCode examples.
Requirements:
- Use the function signature from the interface/code to decide exact parameter names and order.
- Output a strict JSON list. Each item:
  {"name": "ex1", "input": {<param>: <value>, ...}, "expected": <value>}
- Input keys MUST match the function parameters exactly (case-sensitive).
- Keep types natural: strings as strings, ints as ints, arrays as arrays.
- Do not add extra keys. Do not wrap in code fences. Output ONLY JSON."""

    human_prompt = f"""Language: {language}

Interface (may be C++ or Python):
```
{interface}
```

Solution code (reference only; do NOT re-emit):
```
{solution_excerpt}
```

Examples (raw):
```
{examples_text}
```

Return ONLY the JSON list.
"""
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    content = str(resp.content).strip()

    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", content).rstrip("`").strip()

    tests = json.loads(content) 
    if not isinstance(tests, list):
        raise ValueError("LLM did not return a JSON list of tests")
    print("[finished] generating test cases via LLM")
    return tests


def gen_tests_node(state: WFState) -> WFState:
    try:
        tests = examples_to_tests_via_llm(
            examples=state["examples"],
            interface=state["interface"],
            language=state["language"],
            solution_code=state.get("code", "")
        )
        if not isinstance(tests, list):
            return {"tests": []}
        return {"tests": tests}
    except Exception:
        return {"tests": []}


#### generate a test runner using LLM, run the tests, and collect results
def gen_runner_with_llm(language: str,
                        signature: Dict[str, Any],
                        tests: List[Dict[str, Any]],
                        solution_source: str = "") -> str:
    """
    Ask LLM to generate a minimal runner. We provide the current solution source
    as context so the runner can align with the actual method names/signature.
    language: "python" | "cpp"
    signature: (optional for python; recommended for cpp)
    tests: LeetCode-style tests.
    solution_source: content of solution.py / solution.cpp
    """
    llm = get_llm()

    if solution_source:
        head = solution_source[:6000]
        tail = solution_source[-6000:] if len(solution_source) > 12000 else ""
        solution_excerpt = head + ("\n...\n" if tail else "") + tail
    else:
        solution_excerpt = ""

    if language.lower() in ("python", "py"):
        sys_p = """You generate a minimal Python3 test harness (run_tests.py) for a LeetCode-style Solution.
Requirements:
- Import `Solution` from a sibling file 'solution.py' (e.g., `from solution import Solution`).
- Determine the entry method from the provided signature; if empty, introspect Solution to find the single public method.
- Inline the provided tests (JSON) as a Python object.
- First try keyword call with the test input dict; if it raises TypeError (unexpected keyword), fallback to positional call using `inspect.signature` order.
- Print a single JSON list to stdout: {"name":..., "ok": bool, "result": ..., "expected": ...}
- Output ONLY valid Python source code. No prose."""
        human_p = f"""Signature (may be empty):
{json.dumps(signature, ensure_ascii=False)}

Tests JSON:
{json.dumps(tests, ensure_ascii=False)}

Here is the exact solution.py content (reference only; do NOT re-emit it, just import from solution):
```python
{solution_excerpt}
```
"""
    else:
        sys_p = """You generate a minimal C++17 test harness (main.cpp) for a LeetCode-style Solution class.
Requirements:
- A file "solution.cpp" exists. MUST include it via: #include "solution.cpp"
- Use the exact method name and parameter order from the signature.
- Inline the provided tests. Compare:
  * floating-point: |a-b|<=1e-6
  * vectors: element-wise equality
  * pair<int,int>: compare both fields
  * otherwise: operator==
- Print one line per case with OK/FAIL. Return 0 if all pass, else 1.
- Output ONLY valid C++ source code. No prose."""
        human_p = f"""Signature JSON:
{json.dumps(signature, ensure_ascii=False)}

Tests JSON:
{json.dumps(tests, ensure_ascii=False)}

Here is the current solution.cpp content (reference only; do NOT re-emit it, just rely on #include "solution.cpp"):
```cpp
{solution_excerpt}
```
"""
    resp = llm.invoke([SystemMessage(content=sys_p), HumanMessage(content=human_p)])
    code = strip_fences(str(resp.content))

    if language.lower() in ("cpp", "c++"):
        if '#include "solution.cpp"' not in code:
            raise ValueError('Generated C++ runner missing #include "solution.cpp"')
        name = signature.get("name", "")
        if name and (f".{name}(" not in code):
            raise ValueError(f"Generated C++ runner does not call method '{name}'")
    print("[finished] generating test runner via LLM")
    return code

EXEC_TIMEOUT_SEC = 30

#### execute the generated runner, collect results
def run_with_llm_generated_runner(language: str, workdir: str, solution_code: str, runner_code: str) -> Dict[str, Any]:
    language = language.lower()
    if language in ("python", "py"):
        sol_path = os.path.join(workdir, "solution.py")
        run_path = os.path.join(workdir, "run_tests.py")
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution_code.strip() + "\n")
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(runner_code.strip() + "\n")

        try:
            subprocess.check_call([sys.executable, "-m", "py_compile", sol_path, run_path], timeout=EXEC_TIMEOUT_SEC)
        except subprocess.CalledProcessError as e:
            return {"passed": False, "fail_reason": f"py_compile_error: {e}", "raw": ""}

        try:
            out = subprocess.check_output([sys.executable, run_path], stderr=subprocess.STDOUT, text=True, timeout=EXEC_TIMEOUT_SEC)
            try:
                results = json.loads(out.strip())
                passed = all(r.get("ok") for r in results) if isinstance(results, list) else False
                return {"passed": passed, "fail_reason": "" if passed else out, "raw": results}
            except Exception:
                return {"passed": False, "fail_reason": "non-json-output", "raw": out}
        except subprocess.CalledProcessError as e:
            return {"passed": False, "fail_reason": e.output, "raw": e.output}
        except subprocess.TimeoutExpired:
            return {"passed": False, "fail_reason": "timeout", "raw": ""}

    else:
        sol_path = os.path.join(workdir, "solution.cpp")
        main_path = os.path.join(workdir, "main.cpp")
        bin_path = os.path.join(workdir, "a.out")
        portable_headers = (
            "#include <iostream>\n#include <vector>\n#include <string>\n#include <unordered_map>\n"
            "#include <map>\n#include <set>\n#include <queue>\n#include <deque>\n#include <stack>\n"
            "#include <algorithm>\n#include <cmath>\n#include <limits>\n#include <numeric>\n#include <functional>\n"
            "#include <utility>\nusing namespace std;\n"
        )
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(portable_headers + solution_code.strip() + "\n")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(runner_code.strip() + "\n")

        try:
            subprocess.check_call(["g++", "-std=c++17", "-fsyntax-only", main_path], timeout=EXEC_TIMEOUT_SEC)
        except subprocess.CalledProcessError as e:
            return {"passed": False, "fail_reason": f"cpp_syntax_error: {e}", "raw": ""}

        try:
            subprocess.check_call(["g++", "-O2", "-std=c++17", main_path, "-o", bin_path], timeout=EXEC_TIMEOUT_SEC)
        except subprocess.CalledProcessError as e:
            return {"passed": False, "fail_reason": f"compile_error: {e}", "raw": ""}

        try:
            out = subprocess.check_output([bin_path], stderr=subprocess.STDOUT, text=True, timeout=EXEC_TIMEOUT_SEC)
            passed = ("FAIL" not in out) and ("error" not in out.lower())
            return {"passed": passed, "fail_reason": "" if passed else out, "raw": out}
        except subprocess.CalledProcessError as e:
            return {"passed": False, "fail_reason": e.output, "raw": e.output}
        except subprocess.TimeoutExpired:
            return {"passed": False, "fail_reason": "timeout", "raw": ""}

def run_tests_unified(language: str, workdir: str, solution_code: str, tests: List[Dict[str, Any]], signature: Dict[str, Any]) -> Dict[str, Any]:
    runner_code = gen_runner_with_llm(
        language=language,
        signature=signature,
        tests=tests,
        solution_source=solution_code,  
    )
    return run_with_llm_generated_runner(language, workdir, solution_code, runner_code)

def run_tests_node(state: WFState) -> WFState:
    lang = (state["language"] or "").lower()
    wd = state["workdir"]

    signature = {}
    if lang in ("cpp","c++"):
        method = parse_cpp_method_name(state.get("interface",""), default="twoSum")
        signature = {"name": method} 

    try:
        res = run_tests_unified(lang, wd, state["code"], state["tests"], signature)
    except Exception as e:
        res = {"passed": False, "fail_reason": f"runner_generation_failed: {e}", "raw": ""}

    return {"test_result": res}


ANALYZE_SYS = """You are a strict grader.
Return a SINGLE JSON object with EXACT keys:
{"time":"O(...)","space":"O(...)","optimal":"yes|no","reason":"..."}
Rules:
- No prose, no code fences, no additional keys.
- "optimal" must be exactly "yes" or "no".
- If unsure, estimate and explain in "reason".
"""

def _truncate_for_prompt(code: str, limit: int = 6000) -> str:
    if not code:
        return ""
    if len(code) <= limit:
        return code
    head = code[:limit//2]
    tail = code[-limit//2:]
    return head + "\n...\n" + tail

#### analyze the time and space complexity of the generated code using LLM
def analyze_complexity_node(state: WFState) -> WFState:
    llm = get_llm()
    code_snippet = _truncate_for_prompt(state["code"], 12000)
    problem = textwrap.dedent(f"""
    Problem:
    {state['description']}

    Constraints:
    {state['constraints']}

    Code (excerpt):
    {code_snippet}
    """)

    try:
        llm_json = llm.bind(response_format={"type": "json_object"})
        resp = llm_json.invoke([SystemMessage(content=ANALYZE_SYS),
                                HumanMessage(content=problem)])
        comp = json.loads(str(resp.content))
        comp = {
            "time": str(comp.get("time", "")),
            "space": str(comp.get("space", "")),
            "optimal": "yes" if str(comp.get("optimal","")).lower() == "yes" else "no",
            "reason": str(comp.get("reason", "")),
        }
        return {"complexity": comp}
    except Exception:
        pass

    try:
        strict_prompt = ANALYZE_SYS + "\nReturn ONLY the JSON object. No code fences."
        resp = llm.invoke([SystemMessage(content=strict_prompt),
                           HumanMessage(content=problem)])
        text = str(resp.content).strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        comp = json.loads(text)
        comp = {
            "time": str(comp.get("time", "")),
            "space": str(comp.get("space", "")),
            "optimal": "yes" if str(comp.get("optimal","")).lower() == "yes" else "no",
            "reason": str(comp.get("reason", "")),
        }
        return {"complexity": comp}
    except Exception:
        pass

    try:
        m = re.search(r"\{[\s\S]*\}", str(resp.content))
        if m:
            comp = json.loads(m.group(0))
            comp = {
                "time": str(comp.get("time", "")),
                "space": str(comp.get("space", "")),
                "optimal": "yes" if str(comp.get("optimal","")).lower() == "yes" else "no",
                "reason": str(comp.get("reason", "")),
            }
            return {"complexity": comp}
    except Exception:
        pass

    return {"complexity": {"time": "unknown", "space": "unknown", "optimal": "no", "reason": "parse-failed"}}


def summary_report_node(state: WFState) -> WFState:
    report = textwrap.dedent(f"""
    # Solution Summary

    ## Complexity
    - Time: {state['complexity'].get('time')}
    - Space: {state['complexity'].get('space')}
    - Optimal: {state['complexity'].get('optimal')}
    - Notes: {state['complexity'].get('reason')}

    ## Search Used
    - Needed: {state.get('search_needed')}
    - Context (truncated):
```text
{(state.get('search_context') or '')[:1200]}
```

    ## Test Cases (sample)
```json
{json.dumps(state['tests'][:10], ensure_ascii=False, indent=2)}
```

    ## Test Result
```json
{json.dumps(state['test_result'], ensure_ascii=False, indent=2)}
```

    ## Code
```{('cpp' if state['language'] in ('cpp','c++') else 'python')}
{state['code']}
```
    """).strip()

    path = os.path.join(state["workdir"], "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return {"report_path": path}

def after_run_tests(state: WFState) -> str:
    return "gen_code" if not state.get("test_result", {}).get("passed", True) else "analyze_complexity"

def after_analyze_complexity(state: WFState) -> str:
    optimal = str(state.get("complexity", {}).get("optimal", "no")).lower() == "yes"
    return "summary_report" if optimal else "gen_code"

def _dedup_tests(tests):
    """Deduplicate tests by their 'input' JSON (order-insensitive)."""
    seen, out = set(), []
    for t in tests or []:
        try:
            k = json.dumps(t.get("input", {}), sort_keys=True, ensure_ascii=False)
        except Exception:
            k = str(t.get("input", ""))
        if k not in seen:
            seen.add(k); out.append(t)
    return out

#### augment existing test cases using LLM
def augment_tests_node(state: WFState) -> WFState:
    if state.get("augment_tests") is False:
        return {"tests": state.get("tests", [])}

    base = state.get("tests", []) or []

    llm = get_llm()
    sys_prompt = """You generate ADDITIONAL high-coverage unit tests for a LeetCode-style problem.
Return ONLY a JSON object:
{"tests": [{"name":"edge1","input":{...},"expected":...}, ...]}

Rules:
- Respect the interface: parameter names and order must match.
- Derive boundary/degenerate/randomized-but-deterministic cases from Constraints.
- Keep types natural (ints, strings, arrays).
- Do NOT duplicate existing inputs.
- Output ONLY JSON. No code fences."""

    base_preview = base[:20]

    human_prompt = f"""Language: {state.get('language')}

Interface:
```
{state.get('interface','')}
```

Constraints:
```
{state.get('constraints','')}
```

Existing tests (sample, JSON):
{json.dumps(base_preview, ensure_ascii=False)}

Please generate additional tests focusing on boundary/tricky inputs.
Return ONLY the JSON list.
"""
    try:
        try:
            llm_json = llm.bind(response_format={"type": "json_object"})
            resp = llm_json.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
            extra_txt = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception:
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
            extra_txt = str(resp.content)

        txt = extra_txt.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", txt).rstrip("`").strip()

        parsed = json.loads(txt)
        if isinstance(parsed, dict) and "tests" in parsed:
            extra_tests = parsed["tests"]
        else:
            extra_tests = parsed
        if not isinstance(extra_tests, list):
            raise ValueError("extra tests not a list")
    except Exception:
        extra_tests = []

    merged = _dedup_tests((base or []) + (extra_tests or []))
    return {"tests": merged}


#### Workflow graph construction
def build_graph() -> StateGraph:
    builder = StateGraph(WFState)

    builder.add_node("load_spec", load_spec_node)
    builder.add_node("decide_search", decide_search_node)
    builder.add_node("gen_code", gen_code_node)
    builder.add_node("gen_tests", gen_tests_node)
    builder.add_node("run_tests", run_tests_node)
    builder.add_node("augment_edge_tests", augment_tests_node)
    builder.add_node("analyze_complexity", analyze_complexity_node)
    builder.add_node("summary_report", summary_report_node)

    builder.add_edge(START, "load_spec")
    builder.add_edge("load_spec", "decide_search")
    builder.add_edge("decide_search", "gen_code")
    builder.add_edge("gen_code", "gen_tests")
    #builder.add_edge("gen_tests", "augment_edge_tests")
    #builder.add_edge("augment_edge_tests", "run_tests")
    builder.add_edge("gen_tests", "run_tests")
    builder.add_conditional_edges("run_tests", after_run_tests, {
        "gen_code": "gen_code",
        "analyze_complexity": "analyze_complexity",
    })
    builder.add_conditional_edges("analyze_complexity", after_analyze_complexity, {
        "gen_code": "gen_code",
        "summary_report": "summary_report",
    })
    builder.add_edge("summary_report", END)
    return builder



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_path", required=True, help="Path to local problem spec text file")
    parser.add_argument("--language", default="", help="Target language (auto-detect if omitted): python|cpp")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    base_dir = os.path.join(os.getcwd(), "runs")
    workdir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(workdir, exist_ok=True)

    latest = os.path.join(base_dir, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except IsADirectoryError:
                import shutil; shutil.rmtree(latest)
        os.symlink(workdir, latest)
    except Exception:
        pass
    init_state: WFState = {
        "description": "",
        "examples": [],
        "constraints": "",
        "follow_up": "",
        "interface": "",
        "augment_tests": True,  
        "augment_count": 5,

        "language": args.language.strip().lower(),
        "spec_path": args.spec_path,
        "workdir": workdir,

        "search_needed": False,
        "search_context": "",
        "code": "",
        "tests": [],
        "test_result": {"passed": False, "fail_reason": "init"},
        "complexity": {"time": "", "space": "", "optimal": "no", "reason": ""},
        "report_path": "",
    }

    checkpointer = InMemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    final = graph.invoke(init_state, config={"configurable": {"thread_id": str(uuid.uuid4())}})

    print(f"[OK] Report: {final.get('report_path')}")
    print(f"[OK] Workdir: {workdir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
