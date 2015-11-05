#### Virtual setup
It is recommended that you create a `virtualenv` to run the `tlda` code. These instructions are for `ubuntu 14.04` and is expected to work for most newer versions of `ubuntu`. The `virtualenv` and the requirements can be installed using the following steps.

    sudo pip install virtualenv
    sudo apt-get -y build-dep matplotlib  # then enter you root password
    virtualenv -p python2.7 ~/tlda_venv
    source ~/tlda_venv/bin/activate

Note, in the above, the first command `sudo apt-get -y build-dep matplotlib` installs all the build dependencies for `matplotlib`.

#### Clone the repo:

Cloen the repo:    

    git clone https://github.com/GeoscienceAustralia/TLDA.git
    
This will prompt for your github username and password. Once you have entered them and you have access to this repo, `TLDA` will clone in your current directory. 

Once inside the `virtualenv`, navigate to the `TLDA` code:
    
    cd TLDA # This is where the requirements.txt exists
    pip install -r requirements.txt

#### Inputs required to run the TLDA:
These files are not in repo and you will need access to these files to be able to run the code.

* Shapefiles: Hyeuk to describe.
* glenda_reduced: Hyeuk to describe.
* input: Hyeuk to describe.


#### How to run the TLDA code

Running the TLDA code is simple.
    
    cd TLDA
    python transmission/set_data_current_glenda.py 

#### Run tests
To run tests use either `nose` or `unittest`:
    
    cd TLDA
    python -m unittest discover transmission/tests/
    or
    nosetest
