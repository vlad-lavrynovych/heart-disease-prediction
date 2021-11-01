import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv', header=None,
                 names=["UkrName", "Total pertrol", "a92", "a95", "motorMixed", "diesel", "propane", "engName"], sep=";")
df.dropna(subset = ["engName"], inplace=True)
del df["UkrName"]
df = df[df.engName != "Ukraine"]
df["motorMixed"] = df["motorMixed"].str.replace('$','').str.replace(',', '').astype(float)
df["Total pertrol"] = df["Total pertrol"].str.replace('$','').str.replace(',', '').astype(float)
df["diesel"] = df["diesel"].str.replace('$','').str.replace(',', '').astype(float)
df["propane"] = df["propane"].str.replace('$','').str.replace(',', '').astype(float)
print(df)

kyiv = df[df.engName == "Kyiv city"]
zak = df[df.engName == "Zakarpattya"]
print(kyiv)
print(zak)

plt.style.use('ggplot')
total = df["Total pertrol"].dropna()
regions = df["engName"]
print(regions)
# df.plot(x="regions", y="regions")

df.plot(kind='bar',x='engName',y='Total pertrol', title = "Total petrol")
df.plot(kind='bar',x='engName',y='diesel', title = "diesel")
# df.plot(kind='bar',x='engName',y='propane', title = "propane")

plt.show()

# print(total)
# print(df["engName"].dropna())
# plt.hist(total, weights=regions, kind="bar")
# plt.xlabel("Age")
# plt.ylabel("Number of patients")
# plt.title("Age Distribution Among Patients with a Disease")
# plt.show()