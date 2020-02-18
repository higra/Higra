"""
Adapted from https://github.com/cmberryau/rename_dll licensed under the

GNU General Public License v3.0
"""

"""
Make sure you're running this script in a developer command prompt or dumpbin and lib will not be found
"""


def rename_dll(inputdll, outputdll):
    import subprocess
    import re
    from shutil import copyfile
    import os.path as path

    if not path.exists(inputdll):
        raise ValueError("File not found " + inputdll)

    # dump the dll exports using dumpbin
    process = subprocess.Popen(['dumpbin', '/EXPORTS', inputdll], stdout=subprocess.PIPE)
    out, err = process.communicate()

    # get all the function definitions
    lines = out.splitlines(keepends=False)
    pattern = r'^\s*(\d+)\s+[A-Z0-9]+\s+[A-Z0-9]{8}\s+([^ ]+)'

    library_output = 'EXPORTS \n'

    for line in lines:
        matches = re.search(pattern, line.decode('ascii'))

        if matches is not None:
            # ordinal = matches.group(1)
            function_name = matches.group(2)
            library_output = library_output + function_name + '\n'

    # write the def file
    deffile_name = outputdll[:-4] + '.def'
    libfile_name = outputdll[:-4] + '.lib'
    with open(deffile_name, 'w') as f:
        f.write(library_output)

    process = subprocess.Popen(['lib', '/MACHINE:X64', '/DEF:' + deffile_name], )
    out, err = process.communicate()

    # copy the dll over
    copyfile(inputdll, outputdll)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Renames a dll file, generates new def and lib files')
    parser.add_argument('inputdll', help='input dll')
    parser.add_argument('outputdll', help='output dll')
    args = parser.parse_args()
    rename_dll(args.inputdll, args.outputdll)
