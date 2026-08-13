import pypdf, sys, os

os.makedirs('.extract', exist_ok=True)
for name in sys.argv[1:]:
    r = pypdf.PdfReader(f'papers/{name}')
    out = []
    for i, p in enumerate(r.pages):
        out.append(f'\n\n===== PAGE {i+1}/{len(r.pages)} =====\n\n')
        out.append(p.extract_text())
    base = name.replace('.pdf','')
    with open(f'.extract/{base}.txt', 'w') as f:
        f.write(''.join(out))
    print(f'wrote .extract/{base}.txt ({len(r.pages)} pages)')
