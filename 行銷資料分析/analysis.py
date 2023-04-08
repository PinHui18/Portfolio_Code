import plotly.express as px
from dateutil import parser
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import pandas as pd
import re
import numpy as np
# from progressbar import progressbar 


def move_file(dectect_name, folder_name):
    '''
    dectect_name:
        
    folder_name:
        
    '''    
    # 抓出為【正常模型】的所有檔案名稱
    import os 
    save = []
    for i in os.listdir():
        if dectect_name in i:
            save.append(i)
    
    # save=[i for i in os.listdir() if plot_name2 in i]
    
    # make folder
    ff = [i for i in save if not '.' in i ]
    ff = [i for i in ff if  '（' in i ]
    
    
            
    try:
        os.makedirs(folder_name)
        folder_namenew= folder_name
    
    except:
        
        try:
            os.makedirs(folder_name + '（' +str(0)+'）')
            folder_namenew= folder_name + '（' +str(0)+'）'
        except: 
            
            for i in range(0, 10):
                iinn = [j for j in ff if folder_name + '（' +str(i)+'）'  in j]
                if len(iinn) == 0:
                    os.makedirs(folder_name + '（' +str(i)+'）')
                    folder_namenew =folder_name + '（' +str(i)+'）'
                    break
                
                # break
        
    
    
    # move files to that created folder
    import shutil
    save = [i for i in save if '.' in i ]
    for m in save:
        shutil.move(m, folder_namenew)


def list_fun(label, sales_data_tmp,  seg, ip,code):
    '''
    label : mid_mid
    '''
        
    seg = seg[seg['segmentation']==label]
    g = seg[['會員']].merge(sales_data_tmp, on ='會員')
    # sales_data_tmp['年']= sales_data_tmp['訂單時間'].apply(lambda x: x.year)
    g = seg.merge(sales_data_tmp[['會員', '年']], on ='會員', how = 'left').drop_duplicates()
    mem = pd.DataFrame(g['會員'].value_counts()).reset_index()
    mem.columns = ['會員', '不同年的回購次數']
    g = g.merge(mem,on ='會員', how = 'left')
    g = g.sort_values(['年','不同年的回購次數'], ascending = False)
    
    # ip= ip.split('_')[1]
    
    seg = g[['會員']].merge(sales_data_tmp, on ='會員')
    
    g = g.rename(columns ={'count': '消費次數（忠誠度）', 
                                 '單價':'消費金額（貢獻度）',
                                 '年':'最新購買年份',
                                 'segmentation': '區隔' })
    g['建議邀請序'] = range(1, (len(g)+1))
    
    g = g[['會員', '消費次數（忠誠度）',  '消費金額（貢獻度）',	'不同年的回購次數', '最新購買年份','建議邀請序' ,'區隔']]
    
    # 不同年的回購次數
    g= g.sort_values(['最新購買年份','不同年的回購次數', '消費金額（貢獻度）'], ascending = False)
    g =g.drop_duplicates('會員')
    g['建議邀請序'] = range(1, (len(g)+1))
    g['屬性'] = '高回購'
    
    g.to_csv(code + '會員編號清單_高回購_【'+label+'】_' + ip+'.csv', encoding = 'UTF-8-sig')
    
    
    # 消費金額（貢獻度）
    g= g.sort_values(['消費金額（貢獻度）'], ascending = False)
    g['建議邀請序'] = range(1, (len(g)+1))
    g['屬性'] = '高消費'
    g.to_csv(code + '會員編號清單_消費金額_【'+label+'】_' + ip+'.csv', encoding = 'UTF-8-sig')
    
    seg.to_csv(code + '詳細會員原始資料清單_【' +label+'】_'+ ip+'.csv', encoding = 'UTF-8-sig')
    return g,seg




def high_potential(product_2019_yes_over,product_profit_years,product = 'product'):
        
    potential_prod_df = pd.DataFrame()
    for i in product_2019_yes_over[product]:
        tmp =  product_profit_years[product_profit_years[product] == i]
        tmp_base = tmp.iloc[0:len(tmp)-1].reset_index()
        tmp_base['與上一年差異'] = tmp['利潤'].diff().dropna().reset_index()['利潤']
        tmp_base['成長率'] = tmp_base['與上一年差異'] /tmp_base['利潤']
        
        prod = pd.DataFrame({
    
            'product' :[i],
    
            '平均成長率':[ tmp_base['成長率'].mean() ]
        })
        
        potential_prod_df = pd.concat([prod, potential_prod_df], axis = 0)
    
    potential_prod_df = potential_prod_df.sort_values('平均成長率', ascending= False )
    return potential_prod_df
from sklearn.cluster import AgglomerativeClustering

def loy_con_bind(contribution,loyalty, select_product ):
    
    seg = loyalty.merge(contribution, on = '會員')
    seg['segmentation'] = seg['loyalty_precent_rank_cluster'] + '_' + seg['contribution_precent_rank_cluster']
    print(seg['segmentation'].value_counts())
    
    # 使用交叉分析製作顧客區隔
    seg_matrix = pd.crosstab(seg['contribution_precent_rank_cluster'], 
                             seg['loyalty_precent_rank_cluster'] )
    
    seg_matrix = seg_matrix[['low', 'mid','high']]
    seg_matrix  = seg_matrix.reset_index()
    
    
    seg_matrix = pd.concat([
        seg_matrix[seg_matrix['contribution_precent_rank_cluster']=='high'],
        seg_matrix[seg_matrix['contribution_precent_rank_cluster']=='mid'],
        seg_matrix[seg_matrix['contribution_precent_rank_cluster']=='low'],
    ], axis = 0)
    
    print(seg_matrix)
    # seg_matrix.to_csv(select_product+'_顧客區隔分析.csv', encoding = 'utf-8-sig')
    seg_matrix.to_csv('01會員資料區隔_'+ select_product+'.csv', encoding = 'UTF-8-sig')

    return seg, seg_matrix
# 建立function
def customer_loyalty(data, member = '會員' ,
                     product = 'product',
                     select_product = '產品1',year= None):
    '''
    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    member : 字串, optional
        data裡面的「會員」欄位名稱.
        
    select_product : TYPE, optional
        要選擇做顧客分群的產品. The default is '產品1'.

    Returns
    -------
    loyalty : TYPE
        忠誠度（消費次數）.

    '''
    
    # 我們要根據product來準備進行分羣
    sales_data_tmp = data[data[product] ==select_product ]
    
    if year:
        sales_data_tmp = sales_data_tmp[sales_data_tmp['年']==year]
    
    
    # ----忠誠度：消費次數----
    
    # 忠誠度：消費次數計算
    '''
    因爲一筆資料就是消費一次，所以先創建一個過度參數
    來用groupby計算每一個消費者總體消費次數
    '''
    sales_data_tmp['count'] = 1 
    loyalty = sales_data_tmp.groupby([member])[['count']].sum()
    
    # 分羣演算：階層分群(hierarchical clustering)
    from sklearn.cluster import AgglomerativeClustering
    
    # 以階層式分羣法分成3層
    
    # 定義模型
    model = AgglomerativeClustering(n_clusters=3, 
                                    affinity='euclidean', 
                                    linkage='ward')
    
    # 訓練分群模型
    model.fit(loyalty[['count']])
    
    # 抓出loyalty的標籤
    labels = model.labels_
    loyalty['loyalty_cluster'] =labels 
    
    # 將會員從index變成欄位
    loyalty = loyalty.reset_index()
    
    
    
    # cluster標籤高、中、低忠誠度
    
    # 找出每一個cluster標籤的消費次數程度
    cluster_level = loyalty.groupby('loyalty_cluster', as_index = False)['count'].mean()
    cluster_level = cluster_level.sort_values('count', ascending = False)
    
    # 找出最小與最大的cluster
    max_cluster = cluster_level['loyalty_cluster'].iloc[0]
    min_cluster = cluster_level['loyalty_cluster'].iloc[2]
    loyalty['loyalty_precent_rank_cluster'] = np.where(loyalty['loyalty_cluster']==max_cluster, 'high',
                                                np.where(loyalty['loyalty_cluster']==min_cluster, 'low','mid'))
    
    # 查看cluster標籤高、中、低忠誠度次數
    print(loyalty['loyalty_precent_rank_cluster'].value_counts())
    
    return loyalty



# 建立function
def customer_contribution(data, member = '會員' , 
                          price = '單價',
                          product ='product',
                          select_product = '產品1',year= None):
    '''
    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    member : 字串, optional
        data裡面的「會員」欄位名稱.
        
    price : 字串, optional
        data裡面的「單價」欄位名稱.
        
    select_product : TYPE, optional
        要選擇做顧客分群的產品. The default is '產品1'.

    Returns
    -------
    contribution : TYPE
        忠誠度（消費次數）.

    '''
        
    # 我們要根據product來準備進行分羣
    sales_data_tmp = data[data[product] ==select_product ]
    
    # 貢獻度：消費金額計算
    contribution = sales_data_tmp.groupby([member])[[price]].sum()
    
    # 分羣演算：階層分群(hierarchical clustering)
    
    # 以階層式分羣法分成3層
        
    # 定義模型
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', 
                                    linkage='ward')
    
    # 訓練分群模型
    model.fit(contribution[ [price] ])
    
    # 抓出contribution的標籤
    labels = model.labels_    
    contribution['contribution_cluster'] =labels 
    
    # 將會員從index變成欄位
    contribution = contribution.reset_index()
    
    
    
    # cluster標籤高、中、低忠誠度
    
    # 找出每一個cluster標籤的消費次數程度
    cluster_level = contribution.groupby('contribution_cluster', as_index = False)[price].mean()
    cluster_level = cluster_level.sort_values(price, ascending = False)
    
    # 找出最小與最大的cluster
    max_cluster = cluster_level['contribution_cluster'].iloc[0]
    min_cluster = cluster_level['contribution_cluster'].iloc[2]
    contribution['contribution_precent_rank_cluster'] = np.where(contribution['contribution_cluster']==max_cluster, 'high',
                                                np.where(contribution['contribution_cluster']==min_cluster, 'low','mid'))
    
    # 查看cluster標籤高、中、低貢獻度次數
    print(contribution['contribution_precent_rank_cluster'].value_counts())

    return contribution

def product_contribution(data, year ,product, profit,profit_percent = 0.8):
    '''

    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    year : 日期形式
        舉例：'2019-1-1'.
        
    product : 字串
        data裡面的「產品」欄位名稱.
        
    profit : 字串
        data裡面的「利潤」欄位名稱.
        
    profit_percent : int, optional
        篩選貢獻多少「%」利潤優先分析的產品. The default is 0.8.

    Returns
    -------
    建議優先分析的產品.

    '''
        
    sales_data_2019 = data[ (data['訂單時間'] > parser.parse(year)) ]

    # 產品/貢獻比例：計算每一個產品的利潤總和
    product_profit =  sales_data_2019.groupby('product', as_index = False )['利潤'].sum()
    product_profit = product_profit.sort_values('利潤', ascending = False  )
    
    # 產品的貢獻比
    product_profit['利潤佔比'] = product_profit['利潤'] / product_profit['利潤'].sum()
    product_profit['累計利潤佔比'] = product_profit['利潤佔比'].cumsum()
    
    # 產品比
    product_profit['累計產品次數'] = range(1,len(product_profit)+1)
    product_profit['累計產品佔比'] = product_profit['累計產品次數'] / len(product_profit)
    
    # 四捨五入
    product_profit['累計產品佔比'], product_profit['累計利潤佔比'],product_profit['利潤佔比'] = round(product_profit['累計產品佔比'], 2), round(product_profit['累計利潤佔比'], 2), round(product_profit['利潤佔比'], 2)
    
    # 輸出篩選產品貢獻度（利潤）資料
    product_profit.to_csv('0_產品貢獻度（利潤）表.csv', encoding = 'utf-8-sig')
    
    # 產品/貢獻度比例圖
    import plotly.express as px
    
    fig = px.bar(product_profit, x='product', y='利潤佔比',
                 hover_data=['累計利潤佔比', '累計產品佔比'], color='利潤',
                 text = '累計利潤佔比',
                 height= 400,
                 title='產品/貢獻度比例圖'
                 )
    fig.update_traces( textposition='outside')
    plot(fig, filename= '0_產品貢獻度比例圖.html')
    
    
    # 篩選貢獻80%利潤的產品
    
    product_profit_selected = product_profit[product_profit['累計利潤佔比']<=profit_percent]
    
    import plotly.express as px
    
    fig = px.bar(product_profit_selected, x='product', y='利潤佔比',
                 hover_data=['累計利潤佔比', '累計產品佔比'], color='利潤',
                 text = '累計利潤佔比',
                 height= 600,
                 title='貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'
                 )
    fig.update_traces( textposition='outside')
    plot(fig, filename= '1_' + '貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例圖'+'.html')
    
        
    # 建議優先分析的產品
    product_profit_selected.to_csv( '1_' + '貢獻' + str(profit_percent*100) + '%的' + '產品貢獻度比例表'+'.csv', encoding = 'utf-8-sig')
    
    # 歸納資料
    from analysis import  move_file
    move_file(dectect_name = '產品貢獻', folder_name = '0_產品貢獻度')

    
    # 建議優先分析的產品
    return product_profit_selected


def product_potential(data, product ='product', year=2019 ,top=10):
    '''

    Parameters
    ----------
    data : dataFrame
        要放入的交易資料.
        
    year : 日期形式
        舉例：'2019-1-1'.
        
    product : 字串
        data裡面的「產品」欄位名稱.
        
    top : TYPE, optional
        選擇top多少的潛力商品. The default is 10.

    Returns
    -------
    top的潛力商品.

    '''
        
    # 找出每一個產品每一年的利潤
    data['年'] = data['訂單時間'].dt.year
    product_profit_years = data.groupby(['年', product], as_index = False )['利潤'].sum()
    
    # 篩選2019年有的產品
    product_2019_yes = product_profit_years[product_profit_years['年'] ==year]
    
    # 篩選所有product次數
    product_profit_years[product].value_counts()
    each_product = pd.DataFrame(product_profit_years[product].value_counts())
    each_product = each_product.reset_index()
    each_product = each_product.rename(columns ={product:'次數' ,
                                                'index':product})
    
    # 篩選2019年有的產品且已經存活超過1年以上
    product_2019_yes_over = each_product.merge(product_2019_yes, on = product, how= 'right')
    product_2019_yes_over = product_2019_yes_over[product_2019_yes_over['次數'] > 1 ]
    
    # 找出連續年度高潛力產品
    from analysis import high_potential
    potential_prod_df = high_potential(product_2019_yes_over,product_profit_years,product)

    
    # 輸出篩選產品貢獻度（利潤）資料
    potential_prod_df.to_csv('0_連續年度高潛力產品.csv', encoding = 'utf-8-sig')
    
    # 連續年度高潛力產品 - 平均成長率圖
    import plotly.express as px
    
    fig = px.bar(potential_prod_df, x=product, y='平均成長率',
                 hover_data=[product, '平均成長率'], 
                 color='平均成長率',
                 height= 400,
                 title='連續年度高潛力產品 - 平均成長率圖'
                 )
    plot(fig, filename= '0_連續年度高潛力產品 - 平均成長率圖.html')
    
    
    # 連續年度高潛力產品top10 - 平均成長率圖 
    
    import plotly.express as px
    
    fig = px.bar(potential_prod_df[0:top], x=product, y='平均成長率',
                 hover_data=[product, '平均成長率'], 
                 color='平均成長率',
                 height= 600,
                 title='1_連續年度高潛力產品' + str(top) + ' - 平均成長率圖'
                 )
    fig.update_traces( textposition='outside')
    plot(fig, filename= '1_連續年度高潛力產品' + str(top) + ' - 平均成長率圖.html')
    
    potential_prod_df[0:top].to_csv('1_連續年度高潛力產品' + str(top) + ' - 平均成長率表.csv', encoding = 'utf-8-sig')
    
    # 歸納資料
    from analysis import  move_file
    move_file(dectect_name = '連續年度', folder_name = '1_連續年度高潛力產品')

    
    return potential_prod_df[0:top]



def product_client_list(data,
                        seg,
                        product = 'product', select_product ='產品1',
                        member= '會員',
                        price = '單價'
                        ):
        
    # seg_mid = seg[seg['segmentation']=='mid_mid'] 
    
    sales_data_tmp = data[data[product] ==select_product ]
    
    # 使用merge製作完整顧客清單
    client_seg_list = seg[[member,'segmentation']].merge(sales_data_tmp, on =member)
    
    
    # seg, seg_matrix =  seg_matrix_fun(g) #seg, sematrix
    seg_deep_summary = seg.groupby('segmentation', as_index = False)['count', price].mean()
    seg_deep_summary = round(seg_deep_summary, 1 )
    seg_deep_summary = seg_deep_summary.rename(columns ={'count': '消費次數（忠誠度）', 
                                    price:'消費金額（貢獻度）'})
    
    
    seg_deep_summary.to_csv('02會員區隔指標總覽_'+ select_product+'.csv', encoding = 'UTF-8-sig')
    # seg_matrix.to_csv('01會員資料區隔_'+ select_product+'.csv', encoding = 'UTF-8-sig')
    
    # 高消費與高回購
    from analysis import list_fun
    high_high_list, high_high_seg = list_fun(label = 'high_high', sales_data_tmp = sales_data_tmp,  seg = seg, ip=select_product,code= '03')
    mid_mid_list, mid_mid_seg = list_fun(label= 'mid_mid', sales_data_tmp = sales_data_tmp,  seg = seg, ip=select_product,code= '04')
    low_low_list, low_low_seg = list_fun(label= 'low_low', sales_data_tmp = sales_data_tmp,  seg = seg, ip=select_product,code= '05')
    
    move_file(dectect_name = member, folder_name = '2_會員顧客清單')
    return client_seg_list


def ad_pattern(client_seg_list, select_product ='產品1', 
               select_segment = 'high_high',
               top = 10
               ):
    
    client_seg_list_pattern = client_seg_list[client_seg_list['segmentation']==select_segment]
    client_seg_list_pattern['count'] = 1 
    
    # --------廣告---------
    ad_profit = client_seg_list_pattern.groupby('廣告代號all', as_index = False)[['利潤', 'count']].sum()
    
    # 每次廣告利潤
    ad_profit['每次廣告利潤'] = round(ad_profit['利潤'] /ad_profit['count'],2 )
    
    # 佔總利潤比例
    ad_profit['佔總利潤比例'] = ad_profit['利潤'] /ad_profit['利潤'].sum()
    
    # 廣告總利潤比較圖
    ad_profit = ad_profit.sort_values('利潤', ascending = False)
    
    # ad_profit
    ad_profit.to_csv('廣告利潤表_'+ select_product+ '_'+ select_segment+'.csv', encoding = 'utf-8-sig')
    
    # 加註
    ad_profit['product'] = select_product
    ad_profit['區隔'] = select_segment
    
    fig = px.bar(ad_profit, x='廣告代號all', y='利潤',
                 color='利潤',
                 title='廣告總利潤比較圖'
                 )
    plot(fig, filename= '0_廣告總利潤比較圖_'+ select_product+ '_'+ select_segment+'.html')
    
    
    
    # 廣告總利潤比較圖top10
    ad_profit = ad_profit.sort_values('利潤', ascending = False)
    fig = px.bar(ad_profit[0:top], x='廣告代號all', y='利潤',
                 color='利潤',
                 title='1_廣告總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment
                 )
    plot(fig, filename= '1_廣告總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment+'.html')
    
    
    # 每次廣告利潤比較圖
    ad_profit = ad_profit.sort_values('每次廣告利潤', ascending = False)
    
    # 將次數爲 < PR25 的去除
    ad_profit1 = ad_profit[ad_profit['count'] >= ad_profit['count'].describe()['25%'] ]
    ad_profit1 = ad_profit1.sort_values('每次廣告利潤', ascending = False)
    
    fig = px.bar(ad_profit1, x='廣告代號all', y='每次廣告利潤',
                 color='每次廣告利潤',
                 hover_data=['count'],
                 title='每次廣告利潤比較圖_'+ select_product+ '_'+ select_segment,
                 text = '每次廣告利潤'
                 )
    fig.update_traces(texttemplate='%{text:.2s}',  textposition='outside')
    plot(fig, filename= '2_每次廣告利潤比較圖_'+ select_product+ '_'+ select_segment+'.html')
    
    
    # 最適廣告次數與利潤圖
    # 累計除法 --》 廣告打幾次、畫出曲線
    for i in ad_profit[0:top]['廣告代號all']:
            
        curve = client_seg_list_pattern[client_seg_list_pattern['廣告代號all'] ==i]
        curve = curve.sort_values('訂單時間')
        curve = curve[['利潤','count']].cumsum()
        curve['每次廣告利潤'] = round(curve['利潤'] /curve['count'],2 )
        
        curve_max = curve[curve['每次廣告利潤'] == curve['每次廣告利潤'].max()]
        fig = px.line(curve, x="count", y="每次廣告利潤")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=curve['count'], y=curve["每次廣告利潤"],
                        mode='lines',
                        line=dict(color='royalblue')
                        ))
        
        fig.add_trace(go.Scatter(x=curve_max['count'], y=curve_max["每次廣告利潤"],
                        mode="markers+text",
                        text =  str(curve_max['count'].iloc[0])+'次'+'；'+ '每次廣告利潤 : '+ str(curve_max["每次廣告利潤"].iloc[0]),
                        line=dict(color='red'),
                        textposition="top center"
                        ))
        fig.update_layout(title_text='3_最適廣告次數與利潤圖_'+ i + '_'+ select_product+ '_'+ select_segment)
        
        plot(fig, filename= '3_最適廣告次數與利潤圖_'+ i + '_'+ select_product+ '_'+ select_segment+'.html',auto_open=False)


    # 歸納資料
    move_file(dectect_name =select_product, folder_name = select_product+'_'+select_segment+ '_廣告利潤綜整')



def normal_pattern(client_seg_list , 
                   select_product ='產品1', 
               select_segment = 'mid_mid',
               pattern = '尺寸',
               top = 10):
        
    client_seg_list_pattern = client_seg_list[client_seg_list['segmentation']==select_segment]
    client_seg_list_pattern['count'] = 1 
    
    size_profit = client_seg_list_pattern.groupby(pattern, as_index = False)[['利潤', 'count']].sum()
    
    
    # 廣告總利潤比較圖
    size_profit = size_profit.sort_values('利潤', ascending = False)
    
    size_profit.to_csv(pattern+'利潤表_'+ select_product+ '_'+ select_segment+'.csv', encoding = 'utf-8-sig')
    
    
    # 加註
    size_profit['product'] = select_product
    size_profit['區隔'] = select_segment
    
    fig = px.bar(size_profit, x=pattern, y='利潤',
                 color='利潤',
                 title=pattern+'總利潤比較圖'
                 )
    plot(fig, filename= '0_'+pattern+'總利潤比較圖_'+ select_product+ '_'+ select_segment+'.html')
    
    
    
    # 廣告總利潤比較圖top10
    size_profit = size_profit.sort_values('利潤', ascending = False)
    top =10
    fig = px.bar(size_profit[0:top], x=pattern, y='利潤',
                 color='利潤',
                 title= '1_'+pattern+'總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment
                 )
    plot(fig, filename= '1_'+pattern+'總利潤比較圖_top_'+str(top) +'_'+ select_product+ '_'+ select_segment+'.html')
    
    # 歸納資料
    move_file(dectect_name =pattern, 
              folder_name = select_product+'_'+select_segment+ '_'+  pattern+'利潤綜整')
