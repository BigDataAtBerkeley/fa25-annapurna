import os
import time
import json
import uuid
import datetime
import traceback
import contextlib

import boto3
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC