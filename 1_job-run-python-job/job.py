import requests
import json
from requests.auth import HTTPBasicAuth
import os
import warnings
import base64
import cmlapi

v2_api_key = os.environ["CDSW_APIV2_KEY"]
url_base = os.environ["CDSW_DOMAIN"]

client = cmlapi.default_client(url="https://"+os.environ["CDSW_DOMAIN"], cml_api_key=v2_api_key)
username = os.environ["HADOOP_USER_NAME"]
body = cmlapi.RotateV1KeyRequest()
v1_key = client.rotate_v1_key(body, username).api_key
byte_data = v1_key.encode('utf-8')
encodedBytes = base64.b64encode(byte_data)
v1_api_key = str(encodedBytes, 'utf-8')

# allow unsecure connection and suppress warnings
session = requests.Session()
session.verify = False
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

headers_req = {'Content-Type': 'application/json','Authorization': 'Basic ' + v1_api_key}
url_project = os.environ["CDSW_PROJECT_URL"]

runtimes = session.get("https://"+os.environ["CDSW_DOMAIN"]+"/api/v1/runtimes?includeAll=True", headers=headers_req)
for runtime in runtimes.json()['runtimes']:
  if (runtime['editor'] == 'JupyterLab' and runtime['edition'] == 'Standard' 
  and runtime['kernel'] == 'Python 3.10' and runtime['isLatest'] == True):
    jupyter_runtime = runtime
    break

runtime_body = {"runtimesEnabled": [runtime['id']]}
session.patch(url_project, json.dumps(runtime_body), headers=headers_req)