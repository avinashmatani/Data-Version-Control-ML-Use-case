#!"c:\users\avinash\mlops learning\data version control\dvc_ml_use_case\basic_ml_demo\scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'ftfy==6.0.3','console_scripts','ftfy'
__requires__ = 'ftfy==6.0.3'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('ftfy==6.0.3', 'console_scripts', 'ftfy')()
    )
