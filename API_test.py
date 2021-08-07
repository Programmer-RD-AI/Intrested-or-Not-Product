import matplotlib.pyplot as plt
import numpy as np
import os
import unittest
import requests


class Test(unittest.TestCase):
    def test_send_img(self):
        paths = os.listdir("./data/test_data/")
        new_paths = []
        for path in paths:
            new_paths.append(f"./data/test_data/{path}")
        for path in new_paths:
            files = {"file": open(f"{path}", "rb")}
            result = requests.post("http://192.168.1.9:5000/", files=files)
            break
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
