find . \
  \( -path '*/.*' \
     -o -name '__pycache__' -o -name '.mypy_cache' -o -name '.pytest_cache' -o -name '.ruff_cache' \
     -o -name '.tox' -o -name '.nox' -o -name '.venv' -o -name 'venv' \
     -o -name 'build' -o -name 'dist' -o -name 'cmake-build-*' -o -name 'CMakeFiles' -o -name '_deps' \
     -o -name '*.egg-info' -o -name '.eggs' \
     -o -name '.git' -o -name '.hg' -o -name '.svn' \
     -o -name '.vscode' -o -name '.idea' \
     -o -name 'node_modules' \
     -o -name '.DS_Store' \
     -o -name '*.o' -o -name '*.a' -o -name '*.so' -o -name '*.dylib' -o -name '*.dll' \
     -o -name '*.obj' -o -name '*.pdb' \
  \) -prune -o \
  \( -type d \
     -o -name '*.py' -o -name 'pyproject.toml' -o -name 'setup.cfg' -o -name 'setup.py' \
     -o -name 'requirements*.txt' -o -name 'MANIFEST.in' \
     -o -name 'README*' -o -name 'LICENSE*' -o -name 'NOTICE*' -o -name 'CITATION*' \
     -o -name 'CMakeLists.txt' -o -name '*.cmake' \
     -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
     -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
     -o -name '*.cu' -o -name '*.cuh' \
     -o -name '*.proto' \
  \) -print \
| sed -e 's|^\./||' \
| awk -F/ '{
    indent = "";
    for (i=1; i<NF; i++) indent = indent "  ";
    print indent "└─ " $NF
  }'
