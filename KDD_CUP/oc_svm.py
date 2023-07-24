import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM

columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
           "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
           "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
           "dst_host_srv_rerror_rate", "label"]

columns.remove("service")
df = pd.read_pickle("./kddcup.data.pkl")

novelties = df[df["label"] != "normal."]
normal = df[df["label"] == "normal."]
novelties_normal = df[150000:154045]
novelties = pd.concat([novelties, novelties_normal])

for col in normal.columns:
    if normal[col].dtype == "object":
        encoded = LabelEncoder()
        encoded.fit(normal[col])
        normal[col] = encoded.transform(normal[col])

for col in novelties.columns:
    if novelties[col].dtype == "object":
        encoded2 = LabelEncoder()
        encoded2.fit(novelties[col])
        novelties[col] = encoded2.transform(novelties[col])

for f in range(0, 10):
    normal = normal.iloc[np.random.permutation(len(normal))]

df2 = pd.concat([normal[:100000], normal[200000:250000]])
df_validate = normal[100000: 150000]

x_train, x_test = train_test_split(df2.drop("label", axis=1), test_size=0.2, random_state=42)
x_val = df_validate.drop("label", axis=1)

ocsvm = OneClassSVM(kernel='rbf', gamma=0.00005, nu=0.1, verbose=True)
ocsvm.fit(x_train)

preds = ocsvm.predict(x_test)
score = 0
for f in range(0, x_test.shape[0]):
    if preds[f] == 1:
        score += 1

accuracy = score / x_test.shape[0]
print("Accuracy: {:.2%}".format(accuracy))