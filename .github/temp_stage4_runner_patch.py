from pathlib import Path
import re

workflow = Path('.github/workflows/temp-texture-stage4.yml').read_text()
body = workflow.split("cat > /tmp/stage4.py <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
script = '\n'.join(line[10:] if line.startswith('          ') else line for line in body.splitlines())
script = script.replace('from pathlib import Path\n', 'from pathlib import Path\nimport re\n', 1)

old_helper = """def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f'missing marker in {path}: {old[:100]!r}')
    p.write_text(text.replace(old, new, 1))"""
new_helper = """def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old in text:
        p.write_text(text.replace(old, new, 1))
        return
    tokens = [re.escape(token) for token in re.split(r'\\s+', old.strip()) if token]
    pattern = r'\\s+'.join(tokens)
    updated, count = re.subn(pattern, lambda _match: new, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f'missing marker in {path}: {old[:100]!r}')
    p.write_text(updated)"""
if old_helper not in script:
    raise SystemExit('Could not replace Stage 4 patch helper')
script = script.replace(old_helper, new_helper, 1)

start = script.index("replace(cpp, '''        settings.output_white < settings.output_black)")
end = script.index("\nreplace(cpp, '''        {\"maxSize\"", start)
fixed = "replace(cpp, 'settings.output_white < settings.output_black)', 'settings.output_white < settings.output_black || !std::isfinite(settings.mip_sharpen) || settings.mip_sharpen < 0.0f || settings.mip_sharpen > 2.0f || !std::isfinite(settings.deband_strength) || settings.deband_strength < 0.0f || settings.deband_strength > 1.0f)')"
script = script[:start] + fixed + script[end:]
Path('/tmp/stage4.py').write_text(script)
