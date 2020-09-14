import os
import sys
import subprocess
import time
# # os.system("cd ..")
# os.chdir("C:\\Users\\Hong\\openpose")
# # os.system("C:\\Users\\Hong\\openpose\\windows\\x64\Release\\OpenPoseDemo.exe")
# cmd = "C:\\Users\\Hong\\openpose\\windows\\x64\Release\\OpenPoseDemo.exe"
# p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
# time.sleep(5) # <-- sleep for 12''
#
# p.kill()

import subprocess, os, signal
import os
import time
import signal
# def kill(proc_pid):
#     process = psutil.Process(proc_pid)
#     for proc in process.children(recursive=True):
#         proc.kill()
#     process.kill()

# os.chdir("C:\\Users\\Hong\\openpose")
# proc = subprocess.Popen("C:\\Users\\Hong\\openpose\\windows\\x64\Release\\OpenPoseDemo.exe", shell=True)
#
# time.sleep(10)
# # proc.kill(proc.pid)
# # os.kill(proc.pid, signal.CTRL_C_EVENT)
# path = "C:\\Users\\Hong\\openpose\\캡스톤\\result\\*.json"
# os.remove(path)

import glob
import os
directory='C:\\Users\\Hong\\openpose\\캡스톤\\result\\'
os.chdir(directory)
files=glob.glob('*.json')
for filename in files:
    os.unlink(filename)