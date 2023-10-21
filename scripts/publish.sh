pip install -U twine
pip install -U wheel
python3 -m pip install --upgrade build
rm -rf ./build
rm ./dist/*
python3 -m build
twine upload dist/*
rm -rf ./build