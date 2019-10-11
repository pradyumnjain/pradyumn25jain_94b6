import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq

def run():

    number_of_pages = 50

    filename = "tv_flipkart1.csv"
    f = open(filename,"w",encoding="utf-8")

    def fun(data,n):
        if len(data) == 0:
            return "n/a"
        else:
            return data[n].text

    for i in range(1,number_of_pages):
        my_url = "https://www.flipkart.com/search?q=tv&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&p%5B%5D=facets.price_range.from%3D25000&p%5B%5D=facets.price_range.to%3DMax&p%5B%5D=facets.brand%255B%255D%3DSamsung&p%5B%5D=facets.brand%255B%255D%3DLG&p%5B%5D=facets.brand%255B%255D%3DSony&p%5B%5D=facets.serviceability%5B%5D%3Dtrue&page={}".format(i)
        uClient = uReq(my_url)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html,"html.parser")
        containers = page_soup.findAll("div",{"class":"_3O0U0u"})
        headers = "product_name,streaming_service,operating_system,resolution,audio_output,refresh_rate ,external_port,price,Rating, confirmed_buyer\n"
        f.write(headers)
        for container in containers:
            if len(container.find_all('li',attrs={'class':'tVe95H'}))>5:
                product_name = fun(container.find_all("div",class_ = "_3wU53n"),0)
                streaming_service = fun(container.find_all('li',attrs={'class':'tVe95H'}),0)
                operating_system = fun(container.find_all('li',attrs={'class':'tVe95H'}),1)
                if "Oper" not in str(operating_system):
                    continue
                resolution = fun(container.find_all('li',attrs={'class':'tVe95H'}),2)
                if "x" not in str(resolution):
                    continue
                audio_output = fun(container.find_all('li',attrs={'class':'tVe95H'}),3)
                if "Speak" not in str(audio_output):
                    continue
                refresh_rate = fun(container.find_all('li',attrs={'class':'tVe95H'}),4)
                if "Hz" not in str(refresh_rate):
                    continue
                external_port = fun(container.find_all('li',attrs={'class':'tVe95H'}),5)
                if "USB" not in str(external_port):
                    continue
                price = fun(container.find_all("div",class_ = "_1vC4OE _2rQ-NK"),0).replace(",","")
                if "₹" not in str(price):
                    continue
                rating = fun(container.find_all("div",class_ = "hGSR34"),0)
                if "." not in str(rating):
                    continue
                confirmed_buyer = fun(container.find_all("span",class_ = "_38sUEc"),0)
                if "Rat" not in str(confirmed_buyer):
                    continue
    #             print(product_name + "," + streaming_service + "," + operating_system + "," + resolution + "," + audio_output + "," + refresh_rate + "," + external_port + "," + price + ","+ rating + "," +  confirmed_buyer)
                f.write(product_name + "," + streaming_service + "," + operating_system + "," + resolution + "," + audio_output + "," + refresh_rate + "," + external_port + "," + price + ","+ rating + "," +  confirmed_buyer)
                f.write("\n")


    # In[270]:


    f.close()

    df=pd.read_csv("tv_flipkart1.csv",error_bad_lines=False)

    #streaming service
    a = []
    for dat in df["streaming_service"]:
        if "Net" in dat:
            a.append(1)
        else:
            a.append(0)
    df["type"] = a
    df.drop(columns = "streaming_service",axis=0,inplace=True)

    #operating system
    df.drop(columns = "operating_system",axis=0,inplace=True)


    a = []
    for dat in df["resolution"]:
        if "res" in dat:
            a.append(False)
        else:
            a.append(True)
    df = df[a]
    res1 = []
    res2 = []
    for dat in df["resolution"]:
        res1.append(dat.split(" ")[-2])
        res2.append(dat.split(" ")[-4])
    df["res1"] = res1
    df["res2"] = res2
    df.drop(columns = "resolution",axis=0,inplace=True)

    #audio
    audio = []
    for dat in df["audio_output"]:
        audio.append(dat.split(" ")[0])
    df["audio_output"] = audio

    #refresh rate
    r_rate = []
    for dat in df["refresh_rate "]:
        r_rate.append(dat.split(" ")[0])
    df["refresh_rate "] = r_rate

    #external port
    usb_port = []
    hdmi_port = []
    for dat in df["external_port"]:
        usb_port.append(dat.split("|")[1].split(" ")[1])
        hdmi_port.append(dat.split("|")[0].split(" ")[0])
    df["usb_port"] = usb_port
    df["hdmi_port"] = hdmi_port
    df.drop(columns = "external_port",axis=0,inplace=True)

    #rating
    rat_lis = [float(i) for i in df["Rating"]]
    df["Rating"] = rat_lis

    #number of buyer and reviews
    buyers = []
    review = []
    for dat in df[" confirmed_buyer"]:
        buyers.append(dat.split(" ")[0])
        review.append(dat.split(" ")[-2][-1:])
    df["buyers"] = buyers
    df["review"] = review
    df.drop(columns = " confirmed_buyer",axis=0,inplace=True)

    #final price
    price = []
    for dat in df["price"]:
        price.append(dat.strip("₹"))
    df["price"] = price

    return df
