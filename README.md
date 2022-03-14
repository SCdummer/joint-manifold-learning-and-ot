# Template

This is a template for new MIA projects. Using this ensures consistency within the code of the group.

## How to use this template

When you create a new project, select this as template to use. You can do that by
 1. Click on `Create new project`  
 2. Click on `Create from template`
 3. Select the `Group` tab, listed next to the `Built-in` and `Instance` tabs
 4. Unfold `Mathematical Imaging and AI/Templates` 
 5. Click on `Use template` next to this `Machine Learning - Python` template

After you have created a project using this template, go to the issues (see sidebar) and do what is written in the first issue.

## Meaning of the files and folders

| folder/file     | usage |
| ---             | ---   |
| data            | Holds all the data you use; unless it is really little data, *DO NOT* commit the data.
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
 1. What are these `.gitkeep` files?

The template enforces some folder structure. However, git does not care about folders, only about files. So to make sure that git recognizes the folders, a `.gitkeep` is added. Do not remove them. It will confuse git.

 2. What files does git ignore by default and which not?

If you place some file into `data`, then git will ignore them. Please do not remove the `.gitkeep` file in it. Other files created by editors like PyCharm or created when making a virtual env are ignored as well. Most other files are not ignored, and will be committed if you add them to your commits. 

 3. I see no `requirements.txt`. Where is it?

When you work with the package `virtualenv` you use a `requirements.txt` to store you packages. This template assumes that people work with Anaconda. The equivalent to `requirements.txt` for Anaconda is `environment.yml`. If you want to use `virtualenv` and not Anaconda, install the packages from `environment.yml` manually.

 4. I have a question about this template that is not listed in this FAQ. Who do I ask?
 
Ask Tjeerd Jan Heeringa, PhDer with office Zi3006.

