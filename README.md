Hello future me,

in order to update your package, check out this tutorial

https://packaging.python.org/en/latest/tutorials/packaging-projects/


Or in short, run the following commands

'''python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine

python3 -m build
python3 -m twine upload --skip-existing dist/*'''