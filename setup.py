import subprocess
import os

print('---------------------------------------------------')
print('Fisrt steps to execute the Loan Prediction Project.')
print('---------------------------------------------------')
print('')
print('Make sure you have at least Python 3.6 installed and setup a virtual environment.')
print('')
print('Current Python 3 version:')
subprocess.call("python3 --version",shell=True)
print('')

print('Let''s install necessary packages...')
print('')

command = "pip3 install -r requirements.txt"
subprocess.call(command,shell=True) 
print('')
print('')

print('Done! Good luck!')