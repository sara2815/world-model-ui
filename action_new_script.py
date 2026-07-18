# ""
 
 
 
""
 
import argparse
import json
import re
from pathlib import Path
 
CALL_RE = re.compile(r'^\s*(pyautogui\.\w+\(.*\)|time\.sleep\(.*\))\s*$', re.MULTILINE)
COORD_RE = re.compile(r'x\s*=\s*-?\d+(?:\.\d+)?\s*,\s*y\s*=\s*-?\d+(?:\.\d+)?')
RESOLVED_DESC_RE = re.compile(r'\(([^()]+)\)\s*$')
RESOLVED_ITEM_RE = re.compile(r'item\s+(\d+)', re.IGNORECASE)
 
FUNC_RE = re.compile(r'pyautogui\.(\w+)\((.*)\)$')
SLEEP_RE = re.compile(r'time\.sleep\(\s*([\d.]+)\s*\)')
 
# Trailing generic UI-instruction boilerplate to strip off cleaned labels
BOILERPLATE_RE = re.compile(
    r'\s*(Press|Use|Select|Tab to|Click to|Enter to|Space to)\b.*$',
    re.IGNORECASE
)
 
ROLE_TO_WORD = {
    "button": "button", "a": "link", "link": "link", "input": "field",
    "textbox": "field", "textarea": "text box", "select": "dropdown",
    "checkbox": "checkbox", "radio": "radio button", "img": "image",
    "svg": "graphic", "li": "list item", "tab": "tab", "menuitem": "menu item",
    "div": "element", "span": "element", "element": "element",
}
 
MAX_LABEL_WORDS = 10
 
 
def metadata_path_for_step(episode_dir: Path, screenshot_before: str) -> Path:
    base = re.sub(r'(_not_annotated)(_with_cursor)?\.png$', '', screenshot_before)
    return episode_dir / "metadata" / f"{base}_metadata.json"
 
 
def load_metadata_items(episode_dir: Path, screenshot_before: str) -> dict:
    path = metadata_path_for_step(episode_dir, screenshot_before)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {item.get("itemId"): item for item in data.get("annotated_items", [])}
 
 
def describe_from_metadata(item: dict) -> str:
    role = item.get("role") or item.get("tagName") or "element"
    name = item.get("name") or ""
    return f'{role} "{name}"' if name else role
 
 
def raw_label_for(resolved_str: str, meta_items: dict) -> str:
    """Get the raw 'role "name"' string, or '' if nothing informative is available.
    Deliberately does NOT fall back to bare 'item N' -- an unresolved numeric id
    carries no useful signal for the model, so we treat it the same as 'no label'."""
    m = RESOLVED_DESC_RE.search(resolved_str)
    if m:
        return m.group(1).strip()
    m_id = RESOLVED_ITEM_RE.search(resolved_str)
    if m_id:
        item_id = int(m_id.group(1))
        item = meta_items.get(item_id)
        if item:
            return describe_from_metadata(item)
        return ""
    return resolved_str.strip()
 
 
def clean_label(raw_label: str):
    """Turn 'div \"Long verbose label. Press Enter to...\"' into
    ('element', 'Long verbose label')."""
    m = re.match(r'^(\S+)\s+"(.*)"$', raw_label)
    if m:
        role, name = m.group(1), m.group(2)
    else:
        role, name = raw_label, ""
 
    if name:
        name = name.split(". ")[0].split(".\n")[0].strip()
        name = BOILERPLATE_RE.sub("", name).strip().rstrip(".")
        words = name.split()
        if len(words) > MAX_LABEL_WORDS:
            name = " ".join(words[:MAX_LABEL_WORDS])
 
    friendly_role = ROLE_TO_WORD.get(role.lower(), f"{role.lower()} element")
    return friendly_role, name
 
 
def label_phrase(resolved_str: str, meta_items: dict) -> str:
    raw = raw_label_for(resolved_str, meta_items)
    if not raw:
        return "the element"
    role, name = clean_label(raw)
    return f'the {role} "{name}"' if name else f'the {role}'


def coord_str(args_str: str):
    """Extract '(x, y)' as a string from a pyautogui call's args, or None."""
    m = COORD_RE.search(args_str)
    if not m:
        return None
    nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(0))
    return f"({nums[0]}, {nums[1]})"


def coord_fallback_phrase(args_str: str) -> str:
    coords = coord_str(args_str)
    if coords:
        return f"the element at {coords}"
    return "the target element"
 
 
def describe_call(line: str, resolved_str: str, meta_items: dict) -> str:
    """Return a lowercase verb phrase, e.g. 'click the button \"Learn\" located at (324, 528)'."""
    sleep_m = SLEEP_RE.match(line)
    if sleep_m:
        secs = sleep_m.group(1)
        return f"wait {secs} second" + ("" if secs == "1" else "s")
 
    func_m = FUNC_RE.match(line)
    if not func_m:
        return line  # unrecognized line, leave as-is
 
    func, args_str = func_m.group(1), func_m.group(2)
    has_coords = bool(COORD_RE.search(args_str))
    coords = coord_str(args_str) if has_coords else None

    if has_coords:
        # Always start from the descriptive label when available, then append
        # the coordinates onto it -- previously coordinates were dropped
        # whenever a label was available.
        target = label_phrase(resolved_str, meta_items)
        if coords:
            target = f"{target} located at {coords}"
    else:
        target = coord_fallback_phrase(args_str)
 
    if func == "click":
        return f"click {target}"
    if func == "doubleClick":
        return f"double-click {target}"
    if func == "rightClick":
        return f"right-click {target}"
    if func == "tripleClick":
        return f"triple-click {target}"
    if func == "moveTo":
        return f"move the mouse to {target}"
    if func == "dragTo":
        return f"drag to {target}"
    if func == "mouseDown":
        return f"press the mouse button down on {target}"
    if func == "mouseUp":
        return f"release the mouse button on {target}"
    if func == "scroll":
        nums = re.findall(r'-?\d+', args_str)
        amt = int(nums[0]) if nums else 0
        direction = "down" if amt < 0 else "up"
        return f"scroll {direction}" + (f" on {target}" if has_coords else "")
    if func == "hotkey":
        keys = re.findall(r"'([^']+)'|\"([^\"]+)\"", args_str)
        keys = [a or b for a, b in keys]
        return f"press {'+'.join(keys)}" if keys else "press a key combination"
    if func == "press":
        keys = re.findall(r"'([^']+)'|\"([^\"]+)\"", args_str)
        keys = [a or b for a, b in keys]
        return f"press {keys[0]}" if keys else "press a key"
    if func in ("typewrite", "write"):
        text_m = re.search(r"'([^']*)'|\"([^\"]*)\"", args_str)
        text = (text_m.group(1) or text_m.group(2)) if text_m else ""
        return f'type "{text}"'
 
    return f"perform {func}"
 
 
def build_action_new(action_raw: str, actions_resolved: list, meta_items: dict) -> str:
    calls = list(CALL_RE.finditer(action_raw))
    phrases = []
    for call_match, resolved_str in zip(calls, actions_resolved):
        line = call_match.group(1)
        if SLEEP_RE.match(line):
            continue  # sleeps carry no useful signal for the model -- drop them
        phrases.append(describe_call(line, resolved_str, meta_items))
 
    if not phrases:
        return ""
 
    sentence = (", then ".join(phrases)) + "."
    return sentence[0].upper() + sentence[1:]
 
 
def process_file(triplets_path: Path, inplace: bool):
    episode_dir = triplets_path.parent
    data = json.loads(triplets_path.read_text(encoding="utf-8"))
 
    mismatches = 0
    for step in data:
        action_raw = step.get("action_raw", "")
        actions_resolved = step.get("actions_resolved", [])
        screenshot_before = step.get("screenshot_before", "")
        meta_items = load_metadata_items(episode_dir, screenshot_before) if screenshot_before else {}
 
        calls_found = len(CALL_RE.findall(action_raw))
        if calls_found != len(actions_resolved):
            mismatches += 1
 
        step["action_new"] = build_action_new(action_raw, actions_resolved, meta_items)
 
    out_path = triplets_path if inplace else triplets_path.with_name(
        triplets_path.stem + "_new.json"
    )
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    msg = f"Wrote {out_path} ({len(data)} steps)"
    if mismatches:
        msg += f"  [WARNING: {mismatches} step(s) had a call-count/resolved-count mismatch, best-effort alignment used]"
    print(msg)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Root folder to search recursively for actions_triplets.json")
    ap.add_argument("--inplace", action="store_true",
                     help="Overwrite actions_triplets.json instead of writing *_new.json")
    args = ap.parse_args()
 
    root = Path(args.root)
    files = list(root.rglob("actions_triplets.json"))
    if not files:
        print("No actions_triplets.json files found under", root)
        return
 
    for f in files:
        process_file(f, args.inplace)
 
 
if __name__ == "__main__":
    main()
    
    
# import argparse
# import json
# import re
# from pathlib import Path
 
# CALL_RE = re.compile(r'^\s*(pyautogui\.\w+\(.*\)|time\.sleep\(.*\))\s*$', re.MULTILINE)
# COORD_RE = re.compile(r'x\s*=\s*-?\d+(?:\.\d+)?\s*,\s*y\s*=\s*-?\d+(?:\.\d+)?')
# RESOLVED_DESC_RE = re.compile(r'\(([^()]+)\)\s*$')
# RESOLVED_ITEM_RE = re.compile(r'item\s+(\d+)', re.IGNORECASE)
 
# FUNC_RE = re.compile(r'pyautogui\.(\w+)\((.*)\)$')
# SLEEP_RE = re.compile(r'time\.sleep\(\s*([\d.]+)\s*\)')
 
# # Trailing generic UI-instruction boilerplate to strip off cleaned labels
# BOILERPLATE_RE = re.compile(
#     r'\s*(Press|Use|Select|Tab to|Click to|Enter to|Space to)\b.*$',
#     re.IGNORECASE
# )
 
# ROLE_TO_WORD = {
#     "button": "button", "a": "link", "link": "link", "input": "field",
#     "textbox": "field", "textarea": "text box", "select": "dropdown",
#     "checkbox": "checkbox", "radio": "radio button", "img": "image",
#     "svg": "graphic", "li": "list item", "tab": "tab", "menuitem": "menu item",
#     "div": "element", "span": "element", "element": "element",
# }
 
# MAX_LABEL_WORDS = 10
 
 
# def metadata_path_for_step(episode_dir: Path, screenshot_before: str) -> Path:
#     base = re.sub(r'(_not_annotated)(_with_cursor)?\.png$', '', screenshot_before)
#     return episode_dir / "metadata" / f"{base}_metadata.json"
 
 
# def load_metadata_items(episode_dir: Path, screenshot_before: str) -> dict:
#     path = metadata_path_for_step(episode_dir, screenshot_before)
#     if not path.exists():
#         return {}
#     try:
#         data = json.loads(path.read_text(encoding="utf-8"))
#     except Exception:
#         return {}
#     return {item.get("itemId"): item for item in data.get("annotated_items", [])}
 
 
# def describe_from_metadata(item: dict) -> str:
#     role = item.get("role") or item.get("tagName") or "element"
#     name = item.get("name") or ""
#     return f'{role} "{name}"' if name else role
 
 
# def raw_label_for(resolved_str: str, meta_items: dict) -> str:
#     """Get the raw 'role "name"' (or just 'role', or 'item N') string."""
#     m = RESOLVED_DESC_RE.search(resolved_str)
#     if m:
#         return m.group(1).strip()
#     m_id = RESOLVED_ITEM_RE.search(resolved_str)
#     if m_id:
#         item_id = int(m_id.group(1))
#         item = meta_items.get(item_id)
#         if item:
#             return describe_from_metadata(item)
#         return f"item {item_id}"
#     return resolved_str.strip()
 
 
# def clean_label(raw_label: str):
#     """Turn 'div \"Long verbose label. Press Enter to...\"' into
#     ('element', 'Long verbose label')."""
#     m = re.match(r'^(\S+)\s+"(.*)"$', raw_label)
#     if m:
#         role, name = m.group(1), m.group(2)
#     else:
#         role, name = raw_label, ""
 
#     if name:
#         name = name.split(". ")[0].split(".\n")[0].strip()
#         name = BOILERPLATE_RE.sub("", name).strip().rstrip(".")
#         words = name.split()
#         if len(words) > MAX_LABEL_WORDS:
#             name = " ".join(words[:MAX_LABEL_WORDS])
 
#     friendly_role = ROLE_TO_WORD.get(role.lower(), "element")
#     return friendly_role, name
 
 
# def label_phrase(resolved_str: str, meta_items: dict) -> str:
#     raw = raw_label_for(resolved_str, meta_items)
#     role, name = clean_label(raw)
#     return f'the {role} "{name}"' if name else f'the {role}'
 
 
# def coord_fallback_phrase(args_str: str) -> str:
#     m = COORD_RE.search(args_str)
#     if m:
#         nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(0))
#         return f"the element at ({nums[0]}, {nums[1]})"
#     return "the target element"
 
 
# def describe_call(line: str, resolved_str: str, meta_items: dict) -> str:
#     """Return a lowercase verb phrase, e.g. 'click the button \"Learn\"'."""
#     sleep_m = SLEEP_RE.match(line)
#     if sleep_m:
#         secs = sleep_m.group(1)
#         return f"wait {secs} second" + ("" if secs == "1" else "s")
 
#     func_m = FUNC_RE.match(line)
#     if not func_m:
#         return line  # unrecognized line, leave as-is
 
#     func, args_str = func_m.group(1), func_m.group(2)
#     has_coords = bool(COORD_RE.search(args_str))
#     target = label_phrase(resolved_str, meta_items) if has_coords else coord_fallback_phrase(args_str)
 
#     if func == "click":
#         return f"click {target}"
#     if func == "doubleClick":
#         return f"double-click {target}"
#     if func == "rightClick":
#         return f"right-click {target}"
#     if func == "tripleClick":
#         return f"triple-click {target}"
#     if func == "moveTo":
#         return f"move the mouse to {target}"
#     if func == "dragTo":
#         return f"drag to {target}"
#     if func == "mouseDown":
#         return f"press the mouse button down on {target}"
#     if func == "mouseUp":
#         return f"release the mouse button on {target}"
#     if func == "scroll":
#         nums = re.findall(r'-?\d+', args_str)
#         amt = int(nums[0]) if nums else 0
#         direction = "down" if amt < 0 else "up"
#         return f"scroll {direction}" + (f" on {target}" if has_coords else "")
#     if func == "hotkey":
#         keys = re.findall(r"'([^']+)'|\"([^\"]+)\"", args_str)
#         keys = [a or b for a, b in keys]
#         return f"press {'+'.join(keys)}" if keys else "press a key combination"
#     if func == "press":
#         keys = re.findall(r"'([^']+)'|\"([^\"]+)\"", args_str)
#         keys = [a or b for a, b in keys]
#         return f"press {keys[0]}" if keys else "press a key"
#     if func in ("typewrite", "write"):
#         text_m = re.search(r"'([^']*)'|\"([^\"]*)\"", args_str)
#         text = (text_m.group(1) or text_m.group(2)) if text_m else ""
#         return f'type "{text}"'
 
#     return f"perform {func}"
 
 
# def build_action_new(action_raw: str, actions_resolved: list, meta_items: dict) -> str:
#     calls = list(CALL_RE.finditer(action_raw))
#     phrases = []
#     for call_match, resolved_str in zip(calls, actions_resolved):
#         line = call_match.group(1)
#         phrases.append(describe_call(line, resolved_str, meta_items))
 
#     if not phrases:
#         return ""
 
#     sentence = (", then ".join(phrases)) + "."
#     return sentence[0].upper() + sentence[1:]
 
 
# def process_file(triplets_path: Path, inplace: bool):
#     episode_dir = triplets_path.parent
#     data = json.loads(triplets_path.read_text(encoding="utf-8"))
 
#     mismatches = 0
#     for step in data:
#         action_raw = step.get("action_raw", "")
#         actions_resolved = step.get("actions_resolved", [])
#         screenshot_before = step.get("screenshot_before", "")
#         meta_items = load_metadata_items(episode_dir, screenshot_before) if screenshot_before else {}
 
#         calls_found = len(CALL_RE.findall(action_raw))
#         if calls_found != len(actions_resolved):
#             mismatches += 1
 
#         step["action_new"] = build_action_new(action_raw, actions_resolved, meta_items)
 
#     out_path = triplets_path if inplace else triplets_path.with_name(
#         triplets_path.stem + "_new.json"
#     )
#     out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
#     msg = f"Wrote {out_path} ({len(data)} steps)"
#     if mismatches:
#         msg += f"  [WARNING: {mismatches} step(s) had a call-count/resolved-count mismatch, best-effort alignment used]"
#     print(msg)
 
 
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("root", help="Root folder to search recursively for actions_triplets.json")
#     ap.add_argument("--inplace", action="store_true",
#                      help="Overwrite actions_triplets.json instead of writing *_new.json")
#     args = ap.parse_args()
 
#     root = Path(args.root)
#     files = list(root.rglob("actions_triplets.json"))
#     if not files:
#         print("No actions_triplets.json files found under", root)
#         return
 
#     for f in files:
#         process_file(f, args.inplace)
 
 
# if __name__ == "__main__":
#     main()



