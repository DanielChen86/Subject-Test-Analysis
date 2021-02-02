import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit


# AllScores[year] , AllCum[year] , AllCuttings[year]
# plot(obj,fit_with_norm)

# ConvertCum(Scores_cum_now,score_now,Scores_cum_past)

# ConvertCuttings_toCum_Interpol(Scores_cut_now,score_now,Scores_cum_past)
# ConvertCuttingsInterpol(Scores_cut_now,score_now,Scores_cum_past)

# ConvertCuttings_toCum_FitCurve(Scores_cut_now,score_now,Scores_cum_past)
# ConvertCuttingsFitCurve(Scores_cut_now,score_now,Scores_cum_past)

# ConvertCuttings(Scores_cut_now,score_now,Scores_cum_past,method='Interpol')

# plot_compare(sub,now,past,draw_past=False)
# plot_cum(sub,Cut_now,past,draw='both',draw_past=False)

# read file from 'transcript.xlsx'
# transcript , weight , Cut
# total(allscore,weighting={})
# total_converted[year]: total score converted to past

'---------------------------- Read files ----------------------------'



file_num=[100,101,102,103,104,105,106,107,108,109]

file_name=[]
for i in range(len(file_num)):
    file_name.append('./Scores/'+str(file_num[i])+'.xls')


def recording(row_begining,col_begining,df):
    s1=pd.DataFrame(df.iloc[row_begining:row_begining+51,[col_begining,col_begining+1]])
    s1.columns=['score','number']
    s2=pd.DataFrame(df.iloc[row_begining:row_begining+50,[col_begining+7,col_begining+8]])
    s2.columns=['score','number']
    return pd.concat([s1,s2],axis=0,ignore_index=True)


AllScores={}

subjects=['Chinese','English','MathA','MathB','Chemistry','Physics','Biology','History','Geography','Civics']

read_excel_rows=np.arange(3,3+55*len(subjects),55)
for name_num in range(len(file_name)):
    read_score=pd.read_excel(file_name[name_num])
    Scores_temp=pd.DataFrame(index=pd.IntervalIndex(pd.interval_range(0.,101.,101),closed='left')[::-1])
    for i in range(len(subjects)):
        Scores_temp[subjects[i]]=recording(read_excel_rows[i],0,read_score).number.values
    Scores_temp=Scores_temp.astype(int)
    AllScores[file_num[name_num]]=Scores_temp
del read_excel_rows,read_score,file_name,Scores_temp,recording



'---------------------------- Important DataFrames ----------------------------'


AllCum={}
for key in AllScores.keys():
    summation=AllScores[key].sum(axis=0)
    AllCum[key]=(AllScores[key]/summation).cumsum(axis=0)
    AllCum[key].index=AllScores[key].index.left
del summation


def cum_to_cut(cumulation,cutting=[0.12,0.25,0.5,0.75,0.88]):
    result=pd.DataFrame(np.full((len(cutting),len(subjects)),0.),index=cutting,columns=cumulation.columns)
    for sub in subjects:
        Series_temp=cumulation[sub]
        for cut in cutting:
            result[sub][cut]=Series_temp[(Series_temp>cut)].index[0]
    return result


AllCuttings={}
for key,value in AllCum.items():
    AllCuttings[key]=cum_to_cut(value)




def plot(obj,fit_with_norm=False):
    if isinstance(obj,pd.Series):
        if not fit_with_norm:
            plt.plot(range(101,0,-1),obj.values)
            plt.show()
        else:
            data=np.zeros(obj.sum())
            count=0
            for i in range(len(obj)):
                data[count:count+obj.iloc[i]]=np.full(shape=obj.iloc[i],fill_value=obj.index[i].left)
                count+=obj.iloc[i]
            mu,std=st.norm.fit(data)
            plt.plot(range(101,0,-1),obj.values)
            p=st.norm.pdf(np.arange(0,101),mu,std)
            plt.plot(np.arange(0,101),len(data)*p,'k')
            plt.show()
    elif isinstance(obj,pd.DataFrame):
        for col in obj.columns:
            plt.plot(range(101,0,-1),obj[col].values,label=col)
        plt.legend()
        plt.show()



'---------------------------- Convert scores ----------------------------'


# ratio(1,10,3, 20,38)=24.0
def ratio(a_upper,a_lower,a,b_upper,b_lower):
    if isinstance(a,(int,float)):
        if (a_upper==a_lower and b_upper==b_lower):
            return b_upper
        elif (a_upper!=a_lower):
            return (a-a_lower)*(b_upper-b_lower)/(a_upper-a_lower)+b_lower
        else:
            print ("error")
            return None
    elif isinstance(a,pd.Series) or isinstance(a,pd.DataFrame):
        return (a-a_lower)*(b_upper-b_lower)/(a_upper-a_lower)+b_lower




smallest={}
for i in file_num:
    smallest[i]=((AllCuttings[i].shift(1)-AllCuttings[i]).min()).min()
df_smallest=pd.Series(smallest)


# return the translated scores (with respect to the past cumulation of scores), given the cumulation of scores now
def ConvertCum(Scores_cum_now,score_now,Scores_cum_past):
    result={}
    score_after={}
    rank={}
    for key in score_now.keys():
        if (score_now[key]<Scores_cum_now[key].index[0] and score_now[key]>Scores_cum_now[key].index[-1]):
            upper=Scores_cum_now[key].index[Scores_cum_now[key].index>score_now[key]][-1]
            lower=Scores_cum_now[key].index[Scores_cum_now[key].index<=score_now[key]][0]
        elif (score_now[key]>=Scores_cum_now[key].index[0]):
            upper=Scores_cum_now[key].index[0]
            lower=Scores_cum_now[key].index[0]
        elif (score_now[key]<=Scores_cum_now[key].index[-1]):
            upper=Scores_cum_now[key].index[-1]
            lower=Scores_cum_now[key].index[-1]
        else:
            print ("error")
            return {}
        rank[key]=ratio(upper,lower,score_now[key],Scores_cum_now[key][upper],Scores_cum_now[key][lower])
        if (rank[key]>Scores_cum_past[key].iloc[0] and rank[key]<Scores_cum_past[key].iloc[-1]):
            upper_score=Scores_cum_past[key][Scores_cum_past[key]>rank[key]].index[0]
            lower_score=Scores_cum_past[key][Scores_cum_past[key]<=rank[key]].index[-1]
        elif (rank[key]<=Scores_cum_past[key].iloc[0]):
            upper_score=Scores_cum_past[key].index[0]
            lower_score=Scores_cum_past[key].index[0]
        elif (rank[key]>=Scores_cum_past[key].iloc[-1]):
            upper_score=Scores_cum_past[key].index[-1]
            lower_score=Scores_cum_past[key].index[-1]
        else:
            print ("error")
            return {}
        score_after[key]=(upper_score,lower_score)
        result[key]=ratio(Scores_cum_past[key][upper_score],Scores_cum_past[key][lower_score],rank[key],upper_score,lower_score)
    return result


# return the translated cumulation of scores (with respect to the past cumulation of scores), given the cuts of scores now (calculated by interpolation)
def ConvertCuttings_toCum_Interpol(Scores_cut_now,score_now,Scores_cum_past):
    Scores_cut_past=cum_to_cut(cumulation=Scores_cum_past,cutting=Scores_cut_now.index)
    blank_cum=pd.DataFrame(np.full((len(Scores_cum_past),len(score_now)),0.),index=Scores_cum_past.index,columns=score_now.keys())
    screen=pd.DataFrame(np.full((len(Scores_cum_past),len(score_now)),0),index=Scores_cum_past.index,columns=score_now.keys())
    ceiling_now=blank_cum.copy()
    floor_now=blank_cum.copy()
    ceiling_past=blank_cum.copy()
    floor_past=blank_cum.copy()

    for sub in score_now.keys():
        for cut in Scores_cut_now.index:
            screen[sub].loc[Scores_cut_now[sub].loc[cut]]=1
    
    Scores_cum_past_aug=pd.concat([pd.DataFrame(np.full((1,len(Scores_cum_past.columns)),0.),columns=Scores_cum_past.columns,index=[101.]),Scores_cum_past],axis=0)

    Scores_cut_now_aug=pd.concat([pd.DataFrame(np.full((1,len(Scores_cut_now.columns)),101.),columns=Scores_cut_now.columns,index=[0.]),
                                  Scores_cut_now,
                                  pd.DataFrame(np.full((1,len(Scores_cut_now.columns)),0.),columns=Scores_cut_now.columns,index=[100.])])
    Scores_cut_past_aug=pd.concat([pd.DataFrame(np.full((1,len(Scores_cut_now.columns)),101.),columns=Scores_cut_now.columns,index=[0.]),
                                   Scores_cut_past,
                                   pd.DataFrame(np.full((1,len(Scores_cut_now.columns)),0.),columns=Scores_cut_now.columns,index=[100.])])

    screen=screen.cumsum()

    ScoresDataFrame=blank_cum.copy()
    for sub in score_now.keys():
        ScoresDataFrame[sub]=ScoresDataFrame.index

    for sub in score_now.keys():
        Scores_cut_now_dict=Scores_cut_now_aug.set_index(np.arange(len(Scores_cut_now_aug))).to_dict()
        Scores_cut_past_dict=Scores_cut_past_aug.set_index(np.arange(len(Scores_cut_past_aug))).to_dict()
        ceiling_now[sub]=screen[sub].map(Scores_cut_now_dict[sub])
        floor_now[sub]=(screen[sub]+1).map(Scores_cut_now_dict[sub])
        ceiling_past[sub]=screen[sub].map(Scores_cut_past_dict[sub])
        floor_past[sub]=(screen[sub]+1).map(Scores_cut_past_dict[sub])

    relative_scores_pos=ratio(ceiling_now,floor_now,ScoresDataFrame,ceiling_past,floor_past)
    
    floor_relative_pos=relative_scores_pos.apply(np.floor)
    ceiling_relative_pos=floor_relative_pos+1.
    floor_relative_cum=blank_cum.copy()
    ceiling_relative_cum=blank_cum.copy()

    for sub in score_now.keys():
        Scores_cum_past_aug_dict=Scores_cum_past_aug[sub].to_dict()
        floor_relative_cum[sub]=floor_relative_pos[sub].map(Scores_cum_past_aug_dict)
        delta=1e-10
        ceiling_relative_cum[sub]=ceiling_relative_pos[sub].map(Scores_cum_past_aug_dict)+delta
    
    result=ratio(ceiling_relative_pos,floor_relative_pos,relative_scores_pos,ceiling_relative_cum,floor_relative_cum).apply(np.round,args=(8,None))
    return result

# return the translated scores (with respect to the past cumulation of scores), given the cuts of scores now (calculated by interpolation)
def ConvertCuttingsInterpol(Scores_cut_now,score_now,Scores_cum_past):
    cum_temp=(ConvertCuttings_toCum_Interpol(Scores_cut_now,score_now,Scores_cum_past))
    return ConvertCum(cum_temp,score_now,Scores_cum_past)



def polynomial(x,c0,c1,c2,c3,c4):
    # return c0+c1*x+c2*x**2
    return c0+c1*x+c2*x**2+c3*x**3+c4*x**4


def df_full(shape,fill_value,columns,index):
    return (pd.DataFrame(np.full(shape,fill_value),columns=columns,index=index))

# return the translated cumulation of scores (with respect to the past cumulation of scores), given the cuts of scores now (calculated by curve-fitting)
def ConvertCuttings_toCum_FitCurve(Scores_cut_now,score_now,Scores_cum_past):
    Scores_cut_past=cum_to_cut(cumulation=Scores_cum_past,cutting=Scores_cut_now.index)

    Scores_cut_past_aug=pd.concat([df_full((1,len(Scores_cut_now.columns)),101.,Scores_cut_now.columns,[0.]),
                                   Scores_cut_past,
                                   df_full((1,len(Scores_cut_now.columns)),0.,Scores_cut_now.columns,[100.])])
    Scores_cut_now_aug=pd.concat([df_full((1,len(Scores_cut_now.columns)),101.,Scores_cut_now.columns,[0.]),
                                  Scores_cut_now,
                                  df_full((1,len(Scores_cut_now.columns)),0.,Scores_cut_now.columns,[100.])])
    
    blank_cum=pd.DataFrame(np.full((len(Scores_cum_past),len(score_now)),0.),index=Scores_cum_past.index,columns=score_now.keys())
    result=blank_cum.copy()
    screen=pd.DataFrame(np.full((len(Scores_cum_past),len(score_now)),0),index=Scores_cum_past.index,columns=score_now.keys())
    ceiling_now=blank_cum.copy()
    floor_now=blank_cum.copy()
    ceiling_past=blank_cum.copy()
    floor_past=blank_cum.copy()

    Scores_cum_past_aug=pd.concat([pd.DataFrame(np.full((1,len(Scores_cum_past.columns)),0.),columns=Scores_cum_past.columns,index=[101.]),Scores_cum_past],axis=0)

    for sub in score_now.keys():
        for cut in Scores_cut_now.index:
            screen[sub].loc[Scores_cut_now[sub].loc[cut]]=1


    screen=screen.cumsum()
    Scores_cut_now_dict=Scores_cut_now_aug.set_index(np.arange(len(Scores_cut_now_aug))).to_dict()
    Scores_cut_past_dict=Scores_cut_past_aug.set_index(np.arange(len(Scores_cut_past_aug))).to_dict()

    for sub in score_now.keys():
        
        ceiling_now[sub]=screen[sub].map(Scores_cut_now_dict[sub])
        floor_now[sub]=(screen[sub]+1).map(Scores_cut_now_dict[sub])
        ceiling_past[sub]=screen[sub].map(Scores_cut_past_dict[sub])
        floor_past[sub]=(screen[sub]+1).map(Scores_cut_past_dict[sub])

        for i in range(len(Scores_cut_now_aug)-1):
            result['scores']=result.index
            result['scores']=ratio(ceiling_now[sub],floor_now[sub],result['scores'],ceiling_past[sub],floor_past[sub])

            data=Scores_cum_past_aug[sub].loc[Scores_cut_past_aug[sub].iloc[i]:Scores_cut_past_aug[sub].iloc[i+1]]
            xdata=np.array(data.index)
            ydata=np.array(data.values)

            coefficients=curve_fit(polynomial,xdata,ydata)[0]
            result[sub].loc[Scores_cut_now_aug[sub].iloc[i]:Scores_cut_now_aug[sub].iloc[i+1]]\
                =(result['scores'].loc[Scores_cut_now_aug[sub].iloc[i]:Scores_cut_now_aug[sub].iloc[i+1]]).apply(polynomial,args=tuple(coefficients))
            
    result.drop(['scores'],axis=1,inplace=True)
    result=result[result<0.999].fillna(1.)
    result=result[result>0.001].fillna(0.)
    return result



# return the translated scores (with respect to the past cumulation of scores), given the cuts of scores now (calculated by curve-fitting)
def ConvertCuttingsFitCurve(Scores_cut_now,score_now,Scores_cum_past):
    cum_temp=(ConvertCuttings_toCum_FitCurve(Scores_cut_now,score_now,Scores_cum_past))
    return ConvertCum(cum_temp,score_now,Scores_cum_past)


# return the translated scores (with respect to the past cumulation of scores), given the cuts of scores now
def ConvertCuttings(Scores_cut_now,score_now,Scores_cum_past,method='Interpol'):
    if method=='Interpol':
        return ConvertCuttingsInterpol(Scores_cut_now,score_now,Scores_cum_past)
    elif method=='FitCurve':
        return ConvertCuttingsFitCurve(Scores_cut_now,score_now,Scores_cum_past)
    else:
        print ("error")
        return None


# sub: subject; now: a year (int); past: a year (int)
def plot_compare(sub,now,past,draw_past=False):
    subject_dict={sub:sub}
    x_axis=np.arange(100,-1,-1)
    plt.plot(x_axis,AllCum[now][sub],'k',label='now(unknown)')
    plt.plot(x_axis,ConvertCuttings_toCum_Interpol(AllCuttings[now],subject_dict,AllCum[past])[sub],'c',label='interpol')
    plt.plot(x_axis,ConvertCuttings_toCum_FitCurve(AllCuttings[now],subject_dict,AllCum[past])[sub],'g',label='fit curve')
    if draw_past:
        plt.plot(x_axis,AllCum[past][sub],'b',label='past')
    plt.legend()
    plt.show()

# sub: subject; Cut_now: a cut; past: a year (int)
def plot_cum(sub,Cut_now,past,draw='both',draw_past=False):
    subject_dict={sub:sub}
    x_axis=np.arange(100,-1,-1)

    drawing=[]
    if draw=='both':
        drawing=['Interpol','FitCurve']
    elif draw=='Interpol':
        drawing=[draw]
    elif draw=='FitCurve':
        drawing=[draw]
    else:
        drawing=[]

    plt.plot(Cut_now[sub].values,Cut_now[sub].index,marker='o',linestyle='',color='k',label='Cuts')

    if 'Interpol' in drawing:
        plt.plot(x_axis,ConvertCuttings_toCum_Interpol(Cut_now,subject_dict,AllCum[past])[sub],'c',label='interpol')
    if 'FitCurve' in drawing:
        plt.plot(x_axis,ConvertCuttings_toCum_FitCurve(Cut_now,subject_dict,AllCum[past])[sub],'g',label='fit curve')
    if draw_past:
        plt.plot(x_axis,AllCum[past][sub],'b',label='past')
    plt.legend()
    plt.show()


'---------------------------- Read scores ----------------------------'

AllData=pd.read_excel('transcript.xlsx',index_col='index')
AllData=AllData.astype(float)
transcript=AllData.iloc[0].dropna().to_dict()
weight=AllData.iloc[1].dropna().to_dict()
Cut=AllData.iloc[2:7].dropna(axis=1).apply(np.floor)
Cut.index.name=None
Cut.index=Cut.index.astype(float)
del AllData

def total(allscore,weighting={}):
    if isinstance(allscore,pd.Series):
        if weighting=={}:
            for key in allscore.index:
                weighting[key]=1.
        else:
            for key in allscore.index:
                if ((key in weighting.keys())==False):
                    weighting[key]=0.
        weight_temp=pd.Series(weight)
        return (allscore*weight_temp).sum()
    elif isinstance(allscore,dict):
        if weighting=={}:
            for key in allscore.keys():
                weighting[key]=1.
        else:
            for key in allscore.keys():
                if ((key in weighting.keys())==False):
                    weighting[key]=0.
        result=0.
        for key in allscore.keys():
            result+=allscore[key]*weight[key]
        return result

transcript_converted={}
for year in file_num:
    transcript_converted[year]=pd.Series(ConvertCuttings(Cut,transcript,AllCum[year]))[transcript.keys()]
transcript_converted=pd.DataFrame(transcript_converted)

