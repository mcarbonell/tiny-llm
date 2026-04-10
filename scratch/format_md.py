import re
import codecs

filepath = r'c:\Users\mrcm_\Local\proj\tiny-llm\docs\mentes-artificiales.md'

with codecs.open(filepath, 'r', 'utf-8') as f:
    lines = f.readlines()

new_lines = []
in_block = False

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Check if we should close the block
    if in_block:
        is_new_block = stripped in ['Python', 'text', 'XML', 'JSON', 'bash'] and (i+1 < len(lines) and lines[i+1].strip() == '')
        is_header = bool(re.match(r'^(\d+\.|#|##|###|\d+\.\d+)\s+', stripped))
        # Pseudo-code headers inside the text
        is_pseudo_table = bool(re.match(r'^[A-Z][a-z]+\s[A-Za-z]+\s+\|', stripped))
        
        # If it looks like normal text jumping back out
        # "Problema en LLMs actuales:" is not code
        # "Consecuencias:" is not code
        # "Fase 1: Dataset Sintético" is not code
        is_normal_text = False
        if len(stripped) > 0 and stripped[0].isupper() and ':' in stripped and not stripped.startswith('#') and 'print(' not in stripped and '=' not in stripped:
            # Maybe it's a section like "Fase 1: ..."
            is_normal_text = True
            
        if is_new_block or is_header or (is_normal_text and not line.startswith(' ') and not line.startswith('\t')):
            new_lines.append('```\n\n')
            in_block = False

    # Check if we should open a new block
    if not in_block:
        if stripped in ['Python', 'text', 'XML', 'JSON', 'bash'] and (i+1 < len(lines) and lines[i+1].strip() == ''):
            lang = stripped.lower()
            if lang == 'text': lang = 'text'
            new_lines.append(f'```{lang}\n')
            in_block = True
            continue # Skip adding the 'Python' word itself

    # Add the line if it was not skipped
    if not (in_block and stripped in ['Python', 'text', 'XML', 'JSON', 'bash'] and new_lines and new_lines[-1].startswith('```')):
        # Avoid duplicate language strings
        pass
        
    if in_block and stripped in ['Python', 'text', 'XML', 'JSON', 'bash'] and (i+1 < len(lines) and lines[i+1].strip() == ''):
        continue # Already processed
        
    new_lines.append(line)

if in_block:
    new_lines.append('\n```\n')

# Quick pass to format headers properly to markdown if they are not already
final_lines = []
for line in new_lines:
    if re.match(r'^\d+\.\s+[A-Z]', line):
        final_lines.append('## ' + line)
    elif re.match(r'^\d+\.\d+\s+[A-Z]', line):
        final_lines.append('### ' + line)
    else:
        final_lines.append(line)

with codecs.open(filepath, 'w', 'utf-8') as f:
    f.writelines(final_lines)

print("Markdown formatting complete.")
