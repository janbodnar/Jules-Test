#!/usr/bin/env python3
"""
Complete OpenAI Python Ebook Builder
Generates a comprehensive 100-120 page beginner's guide in Typst format.
"""

import re

def read_source_file(filename):
    """Read a source markdown file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return ""

def convert_md_to_typst(md_content):
    """Convert markdown content to Typst format."""
    lines = md_content.split('\n')
    typst_lines = []
    in_code_block = False
    code_lang = ""
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                code_lang = line.strip()[3:].strip() or 'python'
                in_code_block = True
                typst_lines.append(f'```{code_lang}')
            else:
                in_code_block = False
                typst_lines.append('```')
            continue
            
        if in_code_block:
            typst_lines.append(line)
            continue
            
        # Handle headings
        if line.startswith('# '):
            title = line[2:].strip()
            typst_lines.append(f'\n= {title}\n')
        elif line.startswith('## '):
            title = line[3:].strip()
            typst_lines.append(f'\n== {title}\n')
        elif line.startswith('### '):
            title = line[4:].strip()
            typst_lines.append(f'\n=== {title}\n')
        elif line.startswith('---'):
            typst_lines.append('\n#v(1em)\n')
        else:
            if line.strip():
                typst_lines.append(line)
            else:
                typst_lines.append('')
    
    return '\n'.join(typst_lines)

# Read all source files
print("Reading source files...")
sources = {
    'intro': read_source_file('openai/intro.md'),
    'history': read_source_file('openai/history.md'),
    'models': read_source_file('openai/models.md'),
    'openai': read_source_file('openai/openai.md'),
    'openai2': read_source_file('openai/openai2.md'),
    'article1': read_source_file('openai/openai-article1.md'),
    'article2': read_source_file('openai/openai-article2.md'),
}

print("Building comprehensive ebook...")

# Write file
ebook_content = open('ebook_template.txt', 'r').read()

# Process all sources
ebook_content += convert_md_to_typst(sources['intro']) + "\n\n#pagebreak()\n\n"
ebook_content += convert_md_to_typst(sources['history']) + "\n\n#pagebreak()\n\n"
ebook_content += convert_md_to_typst(sources['models']) + "\n\n#pagebreak()\n\n"
ebook_content += convert_md_to_typst(sources['article1']) + "\n\n#pagebreak()\n\n"
ebook_content += convert_md_to_typst(sources['article2']) + "\n\n#pagebreak()\n\n"
ebook_content += convert_md_to_typst(sources['openai']) + "\n\n#pagebreak()\n\n"
ebook_content += convert_md_to_typst(sources['openai2'])

# Add conclusion
ebook_content += open('ebook_conclusion.txt', 'r').read()

# Write output
with open('openai/openai-ebook.typ', 'w', encoding='utf-8') as f:
    f.write(ebook_content)

# Report
char_count = len(ebook_content)
estimated_pages = char_count // 2500
print(f"\n{'='*60}")
print(f"✓ Complete ebook generated!")
print(f"{'='*60}")
print(f"Characters: {char_count:,}")
print(f"Estimated pages: {estimated_pages}")
print(f"Target: 100-120 pages")
print(f"{'='*60}")
