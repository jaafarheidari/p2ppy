##############Price input##################
#buyers's order volum charts

from IPython import get_ipython;   
get_ipython().magic('reset -sf')
#python script.py 105
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
pi_max=13
pi_min=2.5
gam=.5
n_k=30
###################################Microgrids Optimazations################################
#########################MGs max Powerinputs############################
P1max=101.99
P2max=65.69
P3max=101.98
P4max=98.88
P5max=207.37
P6max=342.07
n_MG=6
wrmg1=101.99
wrmg2=65.69
wrmg3=101.98
wrmg4=0
wrmg5=42.86
wrmg6=160.97
########################Microturbines parameters##########################3
a1=0.035
b1=0.1
c1=240
a2=0.0075
b2=2
c2=240
a3=.0078
b3=1
c3=220
a4=0.0087
b4=120
c4=340
a5=.006
b5=110
c5=400
a6=0.0065
b6=120
c6=560
MG1=np.array([P1max,a1,b1,c1])
MG2=np.array([P2max,a2,b2,c2])
MG3=np.array([P3max,a3,b3,c3])
MG4=np.array([P4max,a4,b4,c4])
MG5=np.array([P5max,a5,b5,c5])
MG6=np.array([P6max,a6,b6,c6])
my_d={'MG1':MG1,'MG2':MG2,'MG3':MG3,'MG4':MG4,'MG5':MG5,'MG6':MG6}
MG_Mturbines_df=pd.DataFrame(my_d,index=['Pmax','a','b','c'],columns=['MG1','MG2','MG3','MG4','MG5','MG6'])
sum_n=MG_Mturbines_df.iloc[0,:]#at the beging Pmax is sum_N
ij=0
n_buyers=3
n_sellers=3
#########################MGs dataFrame#####################3
print(MG_Mturbines_df)
#############################MGs optimaziton#################################3
pin=np.arange(pi_min,pi_max,gam) 

Pf=np.empty([len(pin),n_MG])

for j in (np.arange(n_MG)):
    yi=0
    for pi in pin:
        Pf[yi,j]=(pi-MG_Mturbines_df.iloc[2,j])/2*MG_Mturbines_df.iloc[1,j]
        Pf[yi,j]=Pf[yi,j]*1000
        if Pf[yi,j]>MG_Mturbines_df.iloc[0,j]:
            Pf[yi,j]=MG_Mturbines_df.iloc[0,j]
        else:
            Pf[yi,j]=Pf[yi,j]
        yi=yi+1
        
# craet sell dataframe   
bidoptimmax_powers=pd.DataFrame(Pf, columns=['Seller1', 'Seller2','seller3','buyer1','byer2','buyer3'], index=pin)
print(bidoptimmax_powers)
#sell=Pf[Pf>0]
sell=bidoptimmax_powers.iloc[:,0:3]
buy=bidoptimmax_powers.iloc[:,3:6]
buy=buy*-1


######################################33Fibo levels#############################################33
pi_p=np.empty(1000)
pi_p[0]=np.median(pin)
pi_p=pi_p.astype(int)
n_price=len(pin)
########################Buyersers levels (FIB)+maxmin ls:sellers level and lb: buyers level
max_l=100#maximum  of itration that lb define
ki=0 #iteration number
n_p=9#*ki#fibolevels
 #number of iteration#####
trade_s=np.zeros([n_sellers+3,n_buyers+3,n_k,pi_max+1])
trade_b=np.zeros([n_buyers+3,n_sellers+3,n_k,pi_max+1])
trader_buyer=2000*np.ones(1000)
trader_seller=2000*np.ones(1000)
trader_seller_new = np.array([[0, 0, 0, 0]])
vol_trade=np.zeros(10000)
price_trade=np.zeros(10000)
share_buyer=np.zeros([1,n_buyers])
buy_order_new=np.zeros([5,n_buyers])
share_seller=np.zeros([1,n_sellers])
lb=np.empty([5,n_k+1])
dfb=np.empty(n_k+1)
ls=np.empty([5,n_k+1])
dfs=np.empty(n_k+1)
wrmg=np.empty([6,1])
wrmg[0]=wrmg1/P1max
wrmg[1]=wrmg2/P2max
wrmg[2]=wrmg3/P3max
wrmg[3]=wrmg4/P4max
wrmg[4]=wrmg5/P5max
wrmg[5]=wrmg6/P6max
##########################################################3
for sk,ki in enumerate(np.arange(n_k)):
    #Fiboo levels defenition
    if pi_p[ki]==pi_min:
        lb[:,ki]=pi_min
    else:
        lb[0,ki]=pi_min
        dfb[ki]=pi_p[ki]-pi_min
        lb[1,ki]=np.round(pi_min+dfb[ki]*0.236,2)
        lb[2,ki]=np.round(pi_min+dfb[ki]*0.5,2)
        lb[3,ki]=np.round(pi_min+dfb[ki]*0.618,2)
        lb[4,ki]=pi_p[ki]
        bb=np.empty([5,n_k+1])
        
    ############################Sellersers levels (FIB)+maxmin ls:
    if pi_p[ki]==pi_max:
        ls[:,ki]=pi_max
    else:
        ls[0,ki]=pi_p[ki]
        dfs[ki]=pi_max-pi_p[ki]
        ls[1,ki]=np.round(pi_max-dfs[ki]*0.618,2)
        ls[2,ki]=np.round(pi_max-dfs[ki]*0.5,2)
        ls[3,ki]=np.round(pi_max-dfs[ki]*0.236,2)
        ls[4,ki]=pi_max
        ss=np.empty([5,max_l])
        



    #########################WR updat
    x=np.arange(1,6)
    ep=.0001
    cap_mg=np.empty([n_MG,n_k])
    ##################3
    #######sum_n=seleers_buyers_volume22.sum()#######################################################z

    for  i_i,j_j  in enumerate(np.arange(n_MG)):
        cap_mg[i_i,ki]=MG_Mturbines_df.iloc[0,i_i]-sum_n[i_i]
        if wrmg[i_i]*MG_Mturbines_df.iloc[0,i_i] < cap_mg[i_i,ki]:
            wrmg[i_i]=0
        else:
            wrmg[i_i]=(wrmg[i_i]*MG_Mturbines_df.iloc[0,i_i]-cap_mg[i_i,ki])/sum_n[i_i]

    #######################volum of Buyers levels############################33
    x=np.arange(1,6)
    ep=.0001
    volume_bmg=np.empty([5,3])
    ##########################MG4 buyer
    c=(1+2*(wrmg[3]+2))**x
    c=np.round(c,2)
    lcb4=np.empty(5)
    for  i in np.arange(0,5):
        lcb4[i]=c[i]/sum(c)
    lcb4=np.flip(lcb4)
    volume_bmg[:,0]=lcb4*sum_n[3]
    #####MG5#Buyer
    c=(1+2*(wrmg[4]+2))**x
    c=np.round(c,2)
    lcb5=np.empty(5)
    for  i in np.arange(0,5):
        lcb5[i]=c[i]/sum(c)
    lcb5=np.flip(lcb5)
    volume_bmg[:,1]=lcb5*sum_n[4]
    #####MG5#Buyer
    c=(1+2*(wrmg[5]+2))**x
    c=np.round(c,2)
    lcb6=np.empty(5)
    for  i in np.arange(0,5):
        lcb6[i]=c[i]/sum(c)
    lcb6=np.flip(lcb6)
    volume_bmg[:,2]=lcb6*sum_n[5]
    volume_bmg=np.round(volume_bmg,2)

    #######################volum of sellers levels############################33
    volume_smg=np.empty([5,3])
    ##########################MG1 seller
    c=0
    c=(1+2*(wrmg[0]+2))**-x
    c=np.round(c,2)
    lcs1=np.empty(5)
    for  i in np.arange(0,5):
        lcs1[i]=c[i]/sum(c)
    lcs1=np.flip(lcs1)
    volume_smg[:,0]=lcs1*sum_n[0]
    #####MG2#seller
    c=0
    c=(1+2*(wrmg[1]+2))**-x
    c=np.round(c,2)
    lcs2=np.empty(5)
    for  i in np.arange(0,5):
        lcs2[i]=c[i]/sum(c)
    lcs2=np.flip(lcs2)
    volume_smg[:,1]=lcs2*sum_n[1]
    #####MG3 seller
    c=0
    c=(1+2*(wrmg[2]+2))**-x
    c=np.round(c,2)
    lcs3=np.empty(5)
    for  i in np.arange(0,5):
        lcs3[i]=c[i]/sum(c)
    lcs3=np.flip(lcs3)
    volume_smg[:,2]=lcs3*sum_n[2]
    volume_smg=np.round(volume_smg,2)
    ###########################################3656565656566666666666666666
    #seller_dic={'seller1':volume_s1,'seller2':volume_s2}
    ls[:,ki]=ls[:,ki][::-1]
    lb[:,ki]=lb[:,ki][::-1]
    volume_smg=volume_smg[::-1]
    volume_bmg=volume_bmg[::-1]
    sellers_volum_price=pd.DataFrame(volume_smg,index=ls[:,ki],columns=['seller1','seller2','seller3']) #columns=['B3', 'B4'])
    buyers_price=pd.DataFrame(volume_bmg,index=lb[:,ki],columns=['buyer1','buyer2','buyer3']) #columns=['B3', 'B4'])
        #############Dataframe###########################3
    #buyers_price.sort_index(axis=0,ascending=False,inplace=True)
    #sellers_volum_price.sort_index(axis=0,ascending=False,inplace=True)
    seleers_buyers_volume=pd.concat([sellers_volum_price,buyers_price], axis=0,ignore_index=False)
    seleers_buyers_volume
    #seleers_buyers_volume.to_excel('outputsellerbuyer22.xlsx', index=False)

    seleers_buyers_volume=seleers_buyers_volume.fillna(0)
    seleers_buyers_volume=seleers_buyers_volume.round(2)
    print(seleers_buyers_volume)
    #########seller come bach previous version#########
    #sellers_volum_price.sort_index(axis=0,ascending=True,inplace=True)
    ###########################################3656565656566666666666666666666
    #################################################begining of first order######1111111111111111111111111111111
    random_buyernumbers = np.random.random([1, n_buyers])
    buyer_randoindex = np.argsort(random_buyernumbers, axis=1)
    buyer_randoindex=buyer_randoindex+3
    #####seller random selection method

    ############buye_randoindex[0,0]
    buyers_pricen=buyers_price 
    sellers_volum_pricen=sellers_volum_price
    b_vol=np.empty(n_buyers)
    s_vol=np.empty(n_sellers)
    ########115=0,100=1,90=2,80=3,60=4,40=5,35=6,20=7,10=8################3
    #pi_pp=4
    for buyerr,b in  enumerate(np.arange(n_buyers)):
        ###Buyrer random select
        b_vol=np.zeros(n_buyers)
        b_vol[buyerr]=seleers_buyers_volume.iloc[5,buyer_randoindex[0,b]]
        random_sellernumbers = np.random.random([1, n_sellers])
        seller_randoindex = np.argsort(random_sellernumbers, axis=1)
        #####seller random selection method
        for sellers,s in enumerate(np.arange(n_sellers)):
             ###seller random select

            s_vol=np.zeros(n_sellers)
            s_vol[sellers]=seleers_buyers_volume.iloc[4,seller_randoindex[0,s]]
            if s_vol[sellers] < b_vol[buyerr]:
                trade_s[seller_randoindex[0,s],buyer_randoindex[0,b],ki,pi_p[ki]]=s_vol[sellers]
                trade_b[buyer_randoindex[0,b],seller_randoindex[0,s],ki,pi_p[ki]]=s_vol[sellers]
                seleers_buyers_volume.iloc[5,buyer_randoindex[0,b]]=b_vol[buyerr]-s_vol[sellers]
                seleers_buyers_volume.iloc[4,seller_randoindex[0,s]]=0
                b_vol[buyerr]=b_vol[buyerr]-s_vol[sellers]
            else: 
                trade_s[seller_randoindex[0,s],buyer_randoindex[0,b],ki,pi_p[ki]]= b_vol[buyerr]
                trade_b[buyer_randoindex[0,b],seller_randoindex[0,s],ki,pi_p[ki]]= b_vol[buyerr]
                seleers_buyers_volume.iloc[5,buyer_randoindex[0,b]]=0
                seleers_buyers_volume.iloc[4,seller_randoindex[0,s]]=s_vol[sellers] - b_vol[buyerr]
                break

    cols_seleers_buyers_volume = seleers_buyers_volume.columns
    #cols_sell = sellers_volum_pricen.columns
    sum_orders = seleers_buyers_volume[cols_seleers_buyers_volume].sum(axis=1)           
    orderbook=pd.DataFrame(sum_orders, columns=['Sum of orders'], index=seleers_buyers_volume.index)
    sum_sellorders=sellers_volum_pricen.sum(axis=1)
    sum_buyorders=buyers_pricen.sum(axis=1)
    #sum_sellorders = sellers_volum_pricen[cols_sell].sum(axis=1)
    #orderbook_sell=pd.DataFrame(sum_sellorders, columns=['Sum of sell orders'], index=sellers_volum_pricen.index) 

    #################################################End of first order######111111111111111111111111111111111
    ##################################################beginig of other orders########kkkkkkkkkkkkkkkkkkkkkkkkk

    #share_seller[0,0]=sellers_volum_price.iloc[0,0]/orderbook_sell.iloc[0,0]
    #share_seller[0,1]=sellers_volum_price.iloc[0,1]/orderbook_sell.iloc[0,0]

    selle_order_new=np.zeros([n_price,n_sellers])
    ############### if is ok then buyer volume eq to zero and sellers should increas their price
    if  orderbook.iloc[5,0]==0:
        gam_f=0
        share_seller[0,0]=seleers_buyers_volume.iloc[4,0]/orderbook.iloc[4,0]
        share_seller[0,1]=seleers_buyers_volume.iloc[4,1]/orderbook.iloc[4,0]
        share_seller[0,2]=seleers_buyers_volume.iloc[4,2]/orderbook.iloc[4,0]
        ####################create new sell orders############################
        if sum_sellorders.iloc[4] <= sum_buyorders.iloc[1]:
            selle_order_new[0,0]=share_seller[0,0]*sum_buyorders.iloc[1]
            selle_order_new[0,1]=share_seller[0,1]*sum_buyorders.iloc[1]
            selle_order_new[0,2]=share_seller[0,2]*sum_buyorders.iloc[1]
            selle_order_new[0,0]=min(selle_order_new[0,0],seleers_buyers_volume.iloc[4,0])
            selle_order_new[0,1]=min(selle_order_new[0,1],seleers_buyers_volume.iloc[4,1])
            selle_order_new[0,2]=min(selle_order_new[0,2],seleers_buyers_volume.iloc[4,2])
            gam_f=1

        elif sum_sellorders.iloc[4] <= sum_buyorders.iloc[1]+sum_buyorders.iloc[2]:
            selle_order_new[0,0]=share_seller[0,0]*sum_buyorders.iloc[1]
            selle_order_new[0,1]=share_seller[0,1]*sum_buyorders.iloc[1]
            selle_order_new[0,2]=share_seller[0,2]*sum_buyorders.iloc[1]
            selle_order_new[1,0]=seleers_buyers_volume.iloc[4,0]-selle_order_new[0,0]
            selle_order_new[1,1]=seleers_buyers_volume.iloc[4,1]-selle_order_new[0,1]
            selle_order_new[1,2]=seleers_buyers_volume.iloc[4,2]-selle_order_new[0,2]
            gam_f=2
                 #selle_order_new[1,1]=share_seller[0,1]*sum_buyorders.iloc[2]
                 #dif_1=selle_order_new[0,0]+selle_order_new[1,0]
                 #dif_2=selle_order_new[0,1]+selle_order_new[1,1]
                 #rem1=sellers_volum_price.iloc[0,0]-dif_1
                 #rem2=sellers_volum_price.iloc[0,1]-dif_2
                 #selle_order_new[1,0]=min(selle_order_new[1,0],rem1)
                 #selle_order_new[1,1]=min(selle_order_new[1,1],rem2)
        elif sum_sellorders.iloc[4] <= sum_buyorders.iloc[1]+sum_buyorders.iloc[2]+sum_buyorders.iloc[3]:
            selle_order_new[0,0]=share_seller[0,0]*sum_buyorders.iloc[1]
            selle_order_new[0,1]=share_seller[0,1]*sum_buyorders.iloc[1]
            selle_order_new[0,2]=share_seller[0,2]*sum_buyorders.iloc[1]
            selle_order_new[1,0]=share_seller[0,0]*sum_buyorders.iloc[2]
            selle_order_new[1,1]=share_seller[0,1]*sum_buyorders.iloc[2]
            selle_order_new[1,2]=share_seller[0,2]*sum_buyorders.iloc[2]
                 #selle_order_new[2,0]=share_seller[0,0]*sum_buyorders.iloc[3]
                 #selle_order_new[2,1]=share_seller[0,1]*sum_buyorders.iloc[3]

            dif_1=selle_order_new[0,0]+selle_order_new[1,0]
            dif_2=selle_order_new[0,1]+selle_order_new[1,1]
            dif_3=selle_order_new[0,2]+selle_order_new[1,2]
            selle_order_new[2,0]=seleers_buyers_volume.iloc[4,0]-dif_1
            selle_order_new[2,1]=seleers_buyers_volume.iloc[4,1]-dif_2
            selle_order_new[2,2]=seleers_buyers_volume.iloc[4,2]-dif_3
            gam_f=3
                 #selle_order_new[1,0]=min(selle_order_new[1,0],rem1)
                 #selle_order_new[1,1]=min(selle_order_new[1,1],rem2)

                 #selle_order_new[2,0]=min(selle_order_new[2,0],sellers_volum_price.iloc[0,0])
                 #selle_order_new[2,1]=min(selle_order_new[2,1],sellers_volum_price.iloc[0,1])
        else:
            selle_order_new[0,0]=share_seller[0,0]*sum_buyorders.iloc[1]
            selle_order_new[0,1]=share_seller[0,1]*sum_buyorders.iloc[1]
            selle_order_new[0,2]=share_seller[0,2]*sum_buyorders.iloc[1]
            selle_order_new[1,0]=share_seller[0,0]*sum_buyorders.iloc[2]
            selle_order_new[1,1]=share_seller[0,1]*sum_buyorders.iloc[2]
            selle_order_new[1,2]=share_seller[0,2]*sum_buyorders.iloc[2]
            selle_order_new[2,0]=share_seller[0,0]*sum_buyorders.iloc[3]
            selle_order_new[2,1]=share_seller[0,1]*sum_buyorders.iloc[3]
            selle_order_new[2,2]=share_seller[0,2]*sum_buyorders.iloc[3]
                 #selle_order_new[3,0]=share_seller[0,0]*sum_buyorders.iloc[4]
                 #selle_order_new[3,1]=share_seller[0,1]*sum_buyorders.iloc[4]
            dif_1=selle_order_new[0,0]+selle_order_new[1,0]+selle_order_new[2,0]
            dif_2=selle_order_new[0,1]+selle_order_new[1,1]+selle_order_new[2,1]
            dif_3=selle_order_new[0,2]+selle_order_new[1,2]+selle_order_new[2,2]
            selle_order_new[3,0]=seleers_buyers_volume.iloc[4,0]-dif_1
            selle_order_new[3,1]=seleers_buyers_volume.iloc[4,1]-dif_2
            selle_order_new[3,2]=seleers_buyers_volume.iloc[4,2]-dif_3
            gam_f=4
        #buyers_pricen.sort_index(axis=0,ascending=False,inplace=True)
        #sellers_volum_pricen.sort_index(axis=0,ascending=False,inplace=True)
        seleers_buyers_volume2=seleers_buyers_volume.copy()
        #seleers_buyers_volume2=pd.concat([sellers_volum_pricen,buyers_pricen], axis=0,ignore_index=False)
        #seleers_buyers_volume2=seleers_buyers_volume2.fillna(0)
        seleers_buyers_volume2.iloc[4,:]=0
        seleers_buyers_volume2.iloc[6,0]=selle_order_new[0,0]
        seleers_buyers_volume2.iloc[6,1]=selle_order_new[0,1]
        seleers_buyers_volume2.iloc[6,2]=selle_order_new[0,2]
        seleers_buyers_volume2.iloc[7,0]=selle_order_new[1,0]
        seleers_buyers_volume2.iloc[7,1]=selle_order_new[1,1]
        seleers_buyers_volume2.iloc[7,2]=selle_order_new[1,2]
        seleers_buyers_volume2.iloc[8,0]=selle_order_new[2,0]
        seleers_buyers_volume2.iloc[8,1]=selle_order_new[2,1]
        seleers_buyers_volume2.iloc[8,2]=selle_order_new[2,2]
        seleers_buyers_volume2.iloc[9,0]=selle_order_new[3,0]
        seleers_buyers_volume2.iloc[9,1]=selle_order_new[3,1]
        seleers_buyers_volume2.iloc[9,2]=selle_order_new[3,2]
        print(seleers_buyers_volume2)
        #seleers_buyers_volume2.drop_duplicates(inplace=True)
        
        col_price=seleers_buyers_volume2.index
        seleers_buyers_volume2['price']=col_price
        seleers_buyers_volume2.reset_index(drop=True,inplace=True)
        seleers_buyers_volume22=seleers_buyers_volume2.copy()
        #df2 = df1.copy()
        #seleers_buyers_volume2=seleers_buyers_volume2.round(2)
        ##################################################market clearing process########################################
        b_vol2=np.empty(100)
        s_vol2=np.empty(100)
        ########115=0,100=1,90=2,80=3,60=4,40=5,35=6,20=7,10=8#####################################################################3
        pii_h=np.empty(9)
        pii_hh=np.empty(9)
        pii=5+gam_f
        z=np.arange(5,pii+1)
        z=z[::-1]
        for sss,piiii in enumerate(z):
            random_buyernumbers2 = np.random.random([1, n_buyers])
            buyer_randoindex2 = np.argsort(random_buyernumbers2, axis=1)
            n_buyerr=buyer_randoindex2+3
            piiii_i=seleers_buyers_volume22.iloc[piiii,6]
            piiii_i=piiii_i.astype(int)
            for buyerr,b in  enumerate(np.arange(n_buyers)):
                ###Buyrer random select
                b_vol2=np.zeros(100)
                b_vol2[buyerr]=seleers_buyers_volume22.iloc[piiii,n_buyerr[0,b]]
                random_sellernumbers = np.random.random([1, n_sellers])
                seller_randoindex2 = np.argsort(random_sellernumbers, axis=1)
                #####seller random selection method
                for sellers,s in enumerate(np.arange(n_sellers)):
                     ###seller random select
                    s_vol2=np.zeros(100)
                    s_vol2[sellers]=seleers_buyers_volume22.iloc[piiii,seller_randoindex2[0,s]]
                    if s_vol2[sellers] < b_vol2[buyerr]:
                        trade_s[seller_randoindex2[0,s],buyer_randoindex2[0,b],ki,piiii_i]=s_vol2[sellers]
                        trade_b[buyer_randoindex2[0,b],seller_randoindex2[0,s],ki,piiii_i]=s_vol2[sellers]
                        seleers_buyers_volume22.iloc[piiii,n_buyerr[0,b]]=b_vol2[buyerr]-s_vol2[sellers]
                        seleers_buyers_volume22.iloc[piiii,seller_randoindex2[0,s]]=0
                        b_vol2[buyerr]=b_vol2[buyerr]-s_vol2[sellers]
                    else:   
                        trade_s[seller_randoindex2[0,s],buyer_randoindex2[0,b],ki,piiii_i]= b_vol2[buyerr]
                        trade_b[buyer_randoindex2[0,b],seller_randoindex2[0,s],ki,piiii_i]= b_vol2[buyerr]
                        seleers_buyers_volume22.iloc[piiii,n_buyerr[0,b]]=0
                        seleers_buyers_volume22.iloc[piiii,seller_randoindex2[0,s]]=s_vol2[sellers] - b_vol2[buyerr]
                        break




    ##################else means sell orders equil to zero####################    
    else:
        ###########################################negin of Buyers clearing###bbbbbbbbbbbbbbbbbb
        #Share_buyer[0,0]=buyers_pricen.iloc[0,0]/sum_buyorders.iloc[0]
        gam_f=0
        share_buyer[0,0]=seleers_buyers_volume.iloc[5,3]/orderbook.iloc[5,0]
        share_buyer[0,1]=seleers_buyers_volume.iloc[5,4]/orderbook.iloc[5,0]
        share_buyer[0,2]=seleers_buyers_volume.iloc[5,5]/orderbook.iloc[5,0]
        ####################create new sell orders############################
        if sum_buyorders.iloc[0] <= sum_sellorders.iloc[3]:
            buy_order_new[0,0]=share_buyer[0,0]*sum_sellorders.iloc[3]
            buy_order_new[0,1]=share_buyer[0,1]*sum_sellorders.iloc[3]
            buy_order_new[0,2]=share_buyer[0,2]*sum_sellorders.iloc[3]
            buy_order_new[0,0]=min(buy_order_new[0,0],seleers_buyers_volume.iloc[5,3])
            buy_order_new[0,1]=min(buy_order_new[0,1],seleers_buyers_volume.iloc[5,4])
            buy_order_new[0,2]=min(buy_order_new[0,2],seleers_buyers_volume.iloc[5,5])
            gam_f=1

        elif sum_buyorders.iloc[0] <= sum_sellorders.iloc[3]+sum_sellorders.iloc[2]:
            buy_order_new[0,0]=share_buyer[0,0]*sum_sellorders.iloc[3]
            buy_order_new[0,1]=share_buyer[0,1]*sum_sellorders.iloc[3]
            buy_order_new[0,2]=share_buyer[0,2]*sum_sellorders.iloc[3]
            buy_order_new[1,0]=seleers_buyers_volume.iloc[5,3]-buy_order_new[0,0]
            buy_order_new[1,1]=seleers_buyers_volume.iloc[5,4]-buy_order_new[0,1]
            buy_order_new[1,2]=seleers_buyers_volume.iloc[5,5]-buy_order_new[0,2]
            gam_f=2
                 #selle_order_new[1,1]=share_seller[0,1]*sum_buyorders.iloc[2]
                 #dif_1=selle_order_new[0,0]+selle_order_new[1,0]
                 #dif_2=selle_order_new[0,1]+selle_order_new[1,1]
                 #rem1=sellers_volum_price.iloc[0,0]-dif_1
                 #rem2=sellers_volum_price.iloc[0,1]-dif_2
                 #selle_order_new[1,0]=min(selle_order_new[1,0],rem1)
                 #selle_order_new[1,1]=min(selle_order_new[1,1],rem2)
        elif sum_buyorders.iloc[0] <= sum_sellorders.iloc[3]+sum_sellorders.iloc[2]+sum_sellorders.iloc[1]:
            buy_order_new[0,0]=share_buyer[0,0]*sum_sellorders.iloc[3]
            buy_order_new[0,1]=share_buyer[0,1]*sum_sellorders.iloc[3]
            buy_order_new[0,2]=share_buyer[0,2]*sum_sellorders.iloc[3]
            buy_order_new[1,0]=share_buyer[0,0]*sum_sellorders.iloc[2]
            buy_order_new[1,1]=share_buyer[0,1]*sum_sellorders.iloc[2]
            buy_order_new[1,2]=share_buyer[0,2]*sum_sellorders.iloc[2]
                 #selle_order_new[2,0]=share_seller[0,0]*sum_buyorders.iloc[3]
                 #selle_order_new[2,1]=share_seller[0,1]*sum_buyorders.iloc[3]

            dif_1=buy_order_new[0,0]+buy_order_new[1,0]
            dif_2=buy_order_new[0,1]+buy_order_new[1,1]
            dif_3=buy_order_new[0,2]+buy_order_new[1,2]
            buy_order_new[2,0]=seleers_buyers_volume.iloc[5,3]-dif_1
            buy_order_new[2,1]=seleers_buyers_volume.iloc[5,4]-dif_2
            buy_order_new[2,2]=seleers_buyers_volume.iloc[5,5]-dif_3
            gam_f=3
                 #selle_order_new[1,0]=min(selle_order_new[1,0],rem1)
                 #selle_order_new[1,1]=min(selle_order_new[1,1],rem2)

                 #selle_order_new[2,0]=min(selle_order_new[2,0],sellers_volum_price.iloc[0,0])
                 #selle_order_new[2,1]=min(selle_order_new[2,1],sellers_volum_price.iloc[0,1])
        else:
            buy_order_new[0,0]=share_buyer[0,0]*sum_sellorders.iloc[3]
            buy_order_new[0,1]=share_buyer[0,1]*sum_sellorders.iloc[3]
            buy_order_new[0,2]=share_buyer[0,2]*sum_sellorders.iloc[3]
            buy_order_new[1,0]=share_buyer[0,0]*sum_sellorders.iloc[2]
            buy_order_new[1,1]=share_buyer[0,1]*sum_sellorders.iloc[2]
            buy_order_new[1,2]=share_buyer[0,2]*sum_sellorders.iloc[2]
            buy_order_new[2,0]=share_buyer[0,0]*sum_sellorders.iloc[1]
            buy_order_new[2,1]=share_buyer[0,1]*sum_sellorders.iloc[1]
            buy_order_new[2,2]=share_buyer[0,2]*sum_sellorders.iloc[1]
                 #selle_order_new[3,0]=share_seller[0,0]*sum_buyorders.iloc[4]
                 #selle_order_new[3,1]=share_seller[0,1]*sum_buyorders.iloc[4]
            dif_1=buy_order_new[0,0]+buy_order_new[1,0]+buy_order_new[2,0]
            dif_2=buy_order_new[0,1]+buy_order_new[1,1]+buy_order_new[2,1]
            dif_3=buy_order_new[0,2]+buy_order_new[1,2]+buy_order_new[2,2]
            buy_order_new[3,0]=seleers_buyers_volume.iloc[5,3]-dif_1
            buy_order_new[3,1]=seleers_buyers_volume.iloc[5,4]-dif_2
            buy_order_new[3,2]=seleers_buyers_volume.iloc[5,5]-dif_3
            gam_f=4
        #buyers_pricen.sort_index(axis=0,ascending=False,inplace=True)
        #sellers_volum_pricen.sort_index(axis=0,ascending=False,inplace=True)
        #seleers_buyers_volume2=pd.concat([sellers_volum_pricen,buyers_pricen], axis=0,ignore_index=False)
        seleers_buyers_volume2=seleers_buyers_volume.copy()
        #seleers_buyers_volume2=seleers_buyers_volume2.fillna(0)
        seleers_buyers_volume2.iloc[4,:]=0
        seleers_buyers_volume2.iloc[3,3]=buy_order_new[0,0]
        seleers_buyers_volume2.iloc[3,4]=buy_order_new[0,1]
        seleers_buyers_volume2.iloc[3,5]=buy_order_new[0,2]
        seleers_buyers_volume2.iloc[2,3]=buy_order_new[1,0]
        seleers_buyers_volume2.iloc[2,4]=buy_order_new[1,1]
        seleers_buyers_volume2.iloc[2,5]=buy_order_new[1,2]
        seleers_buyers_volume2.iloc[1,3]=buy_order_new[2,0]
        seleers_buyers_volume2.iloc[1,4]=buy_order_new[2,1]
        seleers_buyers_volume2.iloc[1,5]=buy_order_new[2,2]
        seleers_buyers_volume2.iloc[0,3]=buy_order_new[3,0]
        seleers_buyers_volume2.iloc[0,4]=buy_order_new[3,1]
        seleers_buyers_volume2.iloc[0,5]=buy_order_new[3,2]
        print(seleers_buyers_volume2)
        #seleers_buyers_volume2.drop_duplicates(inplace=True)
        col_price=seleers_buyers_volume2.index
        seleers_buyers_volume2['price']=col_price
        seleers_buyers_volume2.reset_index(drop=True,inplace=True)
        seleers_buyers_volume22=seleers_buyers_volume2.copy()
        #seleers_buyers_volume2=seleers_buyers_volume2.round(2)
        ##################################################market clearing process########################################
        b_vol2=np.empty(100)
        s_vol2=np.empty(100)
        ########115=0,100=1,90=2,80=3,60=4,40=5,35=6,20=7,10=8#####################################################################3
        pii=4-gam_f
        z=np.arange(pii,4)
        #z=z[::-1]
        for sss,piii in enumerate(z):
            random_sellernumbers = np.random.random([1, n_sellers])
            seller_randoindex2 = np.argsort(random_sellernumbers, axis=1)
            piiii_i=seleers_buyers_volume22.iloc[piii,6]
            piiii_i=piiii_i.astype(int) 
            for sellers,s in  enumerate(np.arange(n_sellers)):
                ###Buyrer random select
                s_vol2=np.zeros(100)
                s_vol2[sellers]=seleers_buyers_volume22.iloc[piii,seller_randoindex2[0,s]]
                random_buyernumbers2 = np.random.random([1, n_buyers])
                buyer_randoindex2 = np.argsort(random_buyernumbers2, axis=1)
                n_buyerr=buyer_randoindex2+3
                #####seller random selection method
                for buyeree,b in enumerate(np.arange(n_buyers)):
                     ###seller random select
                    b_vol2=np.zeros(100)
                    b_vol2[buyeree]=seleers_buyers_volume22.iloc[piii,n_buyerr[0,b]]
                    if b_vol2[buyeree] < s_vol2[sellers]:
                        
                        trade_s[seller_randoindex2[0,s],buyer_randoindex2[0,b],ki,piiii_i]=b_vol2[buyeree]
                        trade_b[buyer_randoindex2[0,b],seller_randoindex2[0,s],ki,piiii_i]=b_vol2[buyeree]
                        seleers_buyers_volume22.iloc[piii,seller_randoindex2[0,s]]=s_vol2[sellers]-b_vol2[buyeree]
                        seleers_buyers_volume22.iloc[piii,n_buyerr[0,b]]=0
                        s_vol2[sellers]=s_vol2[sellers]-b_vol2[buyeree]
                    else:   
                        trade_s[seller_randoindex2[0,s],buyer_randoindex2[0,b],ki,piiii_i]= s_vol2[sellers]
                        trade_b[buyer_randoindex2[0,b],seller_randoindex2[0,s],ki,piiii_i]= s_vol2[sellers]
                        seleers_buyers_volume22.iloc[piii,n_buyerr[0,b]]=b_vol2[buyerr]-s_vol2[sellers]
                        seleers_buyers_volume22.iloc[piii,seller_randoindex2[0,s]]=0
                        break

    ######update iteration
    
     ###########################################end of Buyers clearing###bbbbbbbbbbbbbbbbbb

###############################################3
##################################################End of other orders############kkkkkkkkkkkkkkkkkkkkkkkkk

    pi_p[ki+1]=piiii_i
    sum_n=seleers_buyers_volume22.sum()
trader_seller_new=np.delete(trader_seller_new, 0, 0)  
Trade_seller_tobuyer_price_volume=pd.DataFrame(trader_seller_new, columns=['Seeler','Buyer','Price','volum'])
s_ztoall=trader_seller_new[trader_seller_new[:,0]==0]
s_ztoall1=trader_seller_new[trader_seller_new[:,0]==1]
s_ztoall2=trader_seller_new[trader_seller_new[:,0]==2]

s1_b1=s_ztoall[s_ztoall[:,1]==3]
s1_b2=s_ztoall[s_ztoall[:,1]==4]
s1_b3=s_ztoall[s_ztoall[:,1]==5]

s2_b1=s_ztoall1[s_ztoall1[:,1]==3]
s2_b2=s_ztoall1[s_ztoall1[:,1]==4]
s2_b3=s_ztoall1[s_ztoall1[:,1]==5]

s3_b1=s_ztoall2[s_ztoall2[:,1]==3]
s3_b2=s_ztoall2[s_ztoall2[:,1]==4]
s3_b3=s_ztoall2[s_ztoall2[:,1]==5]
Trade_seller1_tobuyer1_price_volume=pd.DataFrame(s1_b1, columns=['Seeler','Buyer','Price','volum'])
Trade_seller1_tobuyer2_price_volume=pd.DataFrame(s1_b2, columns=['Seeler','Buyer','Price','volum'])
Trade_seller1_tobuyer3_price_volume=pd.DataFrame(s1_b3, columns=['Seeler','Buyer','Price','volum'])

Trade_seller2_tobuyer1_price_volume=pd.DataFrame(s2_b1, columns=['Seeler','Buyer','Price','volum'])
Trade_seller2_tobuyer2_price_volume=pd.DataFrame(s2_b2, columns=['Seeler','Buyer','Price','volum'])
Trade_seller2_tobuyer3_price_volume=pd.DataFrame(s2_b3, columns=['Seeler','Buyer','Price','volum'])

Trade_seller3_tobuyer1_price_volume=pd.DataFrame(s3_b1, columns=['Seeler','Buyer','Price','volum'])
Trade_seller3_tobuyer2_price_volume=pd.DataFrame(s3_b2, columns=['Seeler','Buyer','Price','volum'])
Trade_seller3_tobuyer3_price_volume=pd.DataFrame(s3_b3, columns=['Seeler','Buyer','Price','volum'])
s_ztoall=trader_seller_new[trader_seller_new[:,0]==0]
s_ztoall1=trader_seller_new[trader_seller_new[:,0]==1]
s_ztoall2=trader_seller_new[trader_seller_new[:,0]==2]
plt.figure(figsize=(16,17))

s1_b1=s_ztoall[s_ztoall[:,1]==3]
k=[]
k,ddd=s1_b1.shape
k=np.arange(k)
plt.subplot(3,3,1)
plt.plot(k,s1_b1[:,2])
plt.bar(k,s1_b1[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.subplot(3,3,2)
s1_b2=s_ztoall[s_ztoall[:,1]==4]
k=[]
k,ddd=s1_b2.shape
k=np.arange(k)
plt.plot(k,s1_b2[:,2])
plt.bar(k,s1_b2[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.subplot(3,3,3)

s1_b3=s_ztoall[s_ztoall[:,1]==5]
k=[]
k,ddd=s1_b3.shape
k=np.arange(k)
plt.plot(k,s1_b3[:,2])
plt.bar(k,s1_b3[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])



s2_b1=s_ztoall1[s_ztoall1[:,1]==3]
k=[]
k,ddd=s2_b1.shape
k=np.arange(k)
plt.subplot(3,3,4)
plt.plot(k,s2_b1[:,2])
plt.bar(k,s2_b1[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.subplot(3,3,5)
s2_b2=s_ztoall1[s_ztoall1[:,1]==4]
k=[]
k,ddd=s2_b2.shape
k=np.arange(k)
plt.plot(k,s2_b2[:,2])
plt.bar(k,s2_b2[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.subplot(3,3,6)

s2_b3=s_ztoall1[s_ztoall1[:,1]==5]
k=[]
k,ddd=s2_b3.shape
k=np.arange(k)
plt.plot(k,s2_b3[:,2])
plt.bar(k,s2_b3[:,3],color='g')
plt.grid()



s3_b1=s_ztoall2[s_ztoall2[:,1]==3]
k=[]
k,ddd=s3_b1.shape
k=np.arange(k)
plt.subplot(3,3,7)
plt.plot(k,s3_b1[:,2])
plt.bar(k,s3_b1[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.subplot(3,3,8)
s3_b2=s_ztoall2[s_ztoall2[:,1]==4]
k=[]
k,ddd=s3_b2.shape
k=np.arange(k)
plt.plot(k,s3_b2[:,2])
plt.bar(k,s3_b2[:,3],color='g')
plt.grid()
#plt.yticks([100,105,110,115,120])
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.subplot(3,3,9)

s3_b3=s_ztoall2[s_ztoall2[:,1]==5]
k=[]
k,ddd=s3_b3.shape
k=np.arange(k)
plt.plot(k,s3_b3[:,2])
plt.bar(k,s3_b3[:,3],color='g')
plt.grid()
plt.show()



