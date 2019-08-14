import subprocess
from subprocess import CalledProcessError
import os
import tempfile
from pathlib import Path

class ZClient:
    def __init__(self):
        dir_path = Path(__file__).resolve().parent
        self.client_path = str(dir_path / "z3950_client/yaz-client.exe")

    def make_call(self, query):
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                text = 'open tcp:aleph.nkp.cz:9991/AUT-UTF\nquerycharset utf-8\nfind ' + query + '\nshow 1+2\nexit\n'
                tmp.write(text)
            output = subprocess.check_output(self.client_path + " -f " + path, shell=True).decode(encoding="utf-8")
            print(output)
        except CalledProcessError as c:
            print(c.returncode)
            print(c.output)
        finally:
            os.remove(path)


client = ZClient()
client.make_call("hlava")
