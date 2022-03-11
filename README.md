# Template

This is a template for new MIA projects. Using this ensures consistency within the code of the group.

## How to use this template

When you create a new project, select this as template to use. You can do that by
 1. Click on `Create new project`  
 2. Click on `Create from template`
 3. Select the `Group` tab, listed next to the `Built-in` and `Instance` tabs
 4. Unfold `Mathematical Imaging and AI/Templates` 
 5. Click on `Use template` next to this `Machine Learning - Python` template

## Meaning of the files and folders

| folder/file     | usage |
| ---             | ---   |
| data            | Holds all the data you use; unless it is really little data, *DO NOT* commit the data.
| data/raw        | Place to place the raw data. This is the data that you are given, and should not be changed afterwards.
| data/processed  | Place to stored the processed/transformed data. 
| data/temp       | Place to store data when you need a place to store it whilst going from raw data to processed data.
| models          | Place to store your pretrained models.
| notebooks       | Place to store your Jupyter notebooks.
| results         | Place to store the images you produce, notebooks that you export to html and other results.
| src             | Root of where you place all your algorithms and code. Pronounced as `source`.
| src/data        | Folder to store code with which you can download data.
| src/features    | Folder to store code with which you turn the data into something usefull to for your networks.
| src/models      | Folder to store the classes and definitions of the (parts of the) neural networks you are using.
| src/visualizations | Folder to store code that produces your images or graphical simulations.
| src/training    | Folder to store code that trains your neural networks.
| src/utils       | Folder to store code that is used in several of the other parts.
| tests           | Location for code that verifies whether the algorithms you wrote are producing the correct output. A sample test is included.
| .coveragerc     | Configuration for coverage test. You can ignore this file, but do not remove it.
| .flake8         | Configuration for flake8 test. You can ignore this file, but do not remove it.
| .gitignore      | Configuration that describes which files should be included into git. You can ignore this file, but do not remove it.
| .pylintrc       | Configuration for pylint test. You can ignore this file, but do not remove it.
| .gitlab-ci.yml  | Configuration that tells Gitlab how to do the tests automatically on commit. You can ignore this file, but do not remove it.
| README.md       | This is the file that is show when you open the repository online. Use it to explain stuff about your project. If this is for a paper, include a way to cite it.
| environment.yml | Configuration file that is used to create a custom conda environment for the project. Keep it up to date with the packages you use.


## FAQ
 1. I have a question about this template. Who do I ask?
 
Ask Tjeerd Jan Heeringa, PhDer with office Zi3006.
