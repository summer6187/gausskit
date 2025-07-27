# Gausskit
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Documentation style

Gausskit uses [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
docstrings to document its modules, classes, and functions.
***
A user friendly python interface for Gaussian 09/16 and Multiwell packages users.

---

## Compile

Pull this repo from github and install this package with the following command

```shell
git clone git@github.com:summer6187/gausskit.git
cd gausskit
pip install .
```

After this, you will have `gausskit` in your environment. Check it out with `which gausskit`.

Then you have to make sure you have successfully installed your *Multiwell* and  configure your ***multiwell*** commands in `~/.gausskitrc`. Gausskit will read this file `~/.gausskitrc` and understand how to run multiwell in your local machine. An example of `.gausskitrc` is demonstrated in `gausskit/templates/gausskitrc.template`

In Spartan, you will find an example like this

```shell
# System informations to get multiwell running, please adjust:
# When adjusted, please copy to ~/.gausskitrc

[machine]
 gauss2multi_command = module load GCC/11.3.0 OpenMPI/4.1.4; module load Multiwell/2021; echo N | gauss2multi
 parsctst_command = module load GCC/11.3.0 OpenMPI/4.1.4; module load Multiwell/2021; parsctst
 bdens_command = module load GCC/11.3.0 OpenMPI/4.1.4; module load Multiwell/2021; bdens
 thermo_command = module load GCC/11.3.0 OpenMPI/4.1.4; module load Multiwell/2021; thermo
 mominert_command = module load GCC/11.3.0 OpenMPI/4.1.4; module load Multiwell/2021; mominert
 multiwell_command = module load GCC/11.3.0 OpenMPI/4.1.4; module load Multiwell/2021; multiwell
```



## Tutorials

Here are something you can do

```
# 0. show current version of gausskit
gausskit -V
# or gausskit --version

# 1. show the available command you can do with Gausskit
gausskit --help
# you can also show the parameters that you can do with some commands
# for example, gausskit output --help

# 2. postprocess the data and generate database.json
gausskit output <path_to_data_directory_that_has_all_log_files>
# gausskit output <data_dir> --output database.json # this is prefered
# gausskit output <data_dir> --output database.pickle # this will generate pickle database

# 3. check your database
gausskit info <path_to_database.json/database.pickle>

# 4. run thermo
gausskit run PES.in
```

Have a nice day!

