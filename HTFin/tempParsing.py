s = """ Free Brass-1: 30.4335861206055
Free Brass-2: 32.6461601257324
Free Brass-3: 38.6185150146484
Free Brass-4: 51.4693870544434
Free Brass-5: 76.5698852539063
Free Copper-1: 46.2016944885254
Free Copper-2: 50.5833168029785
Free Copper-3: 53.6396980285645
Free Copper-4: 60.3475189208984
Free Copper-5: 68.9532165527344
Free Steel-1: 22.2945384979248
Free Steel-2: 22.5085353851318
Free Steel-3: 24.3150787353516
Free Steel-4: 34.2728729248047
Free Steel-5: 82.7978210449219
Free Aluminum-1: 28.5175647735596
Free Aluminum-2: 29.3290271759033
Free Aluminum-3: 32.6868133544922
Free Aluminum-4: 39.2744026184082
Free Aluminum-5: 51.1504325866699
Forced Brass-1: 23.5697898864746
Forced Brass-2: 24.2858448028564
Forced Brass-3: 26.1337738037109
Forced Brass-4: 32.4907455444336
Forced Brass-5: 57.869441986084
Forced Copper-1: 32.4202079772949
Forced Copper-2: 34.7090759277344
Forced Copper-3: 36.361743927002
Forced Copper-4: 40.8294830322266
Forced Copper-5: 48.8889770507813
Forced Steel-1: 23.0788612365723
Forced Steel-2: 23.0421905517578
Forced Steel-3: 23.4397678375244
Forced Steel-4: 24.6956958770752
Forced Steel-5: 56.0011024475098
Forced Aluminum-1: 25.0694255828857
Forced Aluminum-2: 25.3450527191162
Forced Aluminum-3: 26.7058753967285
Forced Aluminum-4: 29.9052829742432
Forced Aluminum-5: 40.859016418457
"""
L = list(map(lambda x: round(float(x), 4), filter(lambda x: x.count(".") == 1, s.split())))

result = [[L[i*5 + j] for j in range(5)] for i in range(8)]
print(result)