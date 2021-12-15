# si649f21 Group Project -- Robogame

# imports we will use
import json
import os
from networkx.drawing.nx_pydot import graphviz_layout
from itertools import chain
import pydot
import matplotlib.pyplot as plt
import heapq
import numpy as np
import networkx as nx
import Robogame as rg
import pandas as pd
import altair as alt
import streamlit as st
import time
time_start = time.time()

# datamodel imports
pd.options.mode.chained_assignment = None

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

# our key
KEY = "bob"
st.set_page_config(layout='wide')

selectbox = st.sidebar.selectbox(
    label="Select which page to display",
    options=['Visualization', 'Entering Interests'], index=0, key = 'page_select'
)

if 'Visualization' in selectbox:

  # let's create two "stree
  # pots" in the streamlit view for our charts
  col_title, col_time = st.columns((3,3))
  with col_title:
      st.title('SI 649 Robogames -- Team VISION')
  with col_time:
      current_game_time = st.empty()
      current_game_time.markdown(f'## Current game time: {0}')
  st.header('Time-Value Plot for Top 5 Robots')
  col1, col2, col3 = st.columns((5,1,1))
  table_time =pd.DataFrame()

  with col1:
      timevis1 = st.empty()
  with col3:
      st.text('Robots about to expire:')
      table_t = st.table(table_time)
  st.header('Family Tree with Predicted Productivity')
  timevis2 = st.empty()
  st.header('Robot Productivity')
  col21, col22, col23 = st.columns((5,1,1))
  with col21:
      smvis = st.empty()
  with col23:
      boxPlot = st.empty()
st.header('Recommended Hacker Interests')



# constants part list
partList = ['Repulsorlift Motor HP', 'Astrogation Buffer Length', 'Polarity Sinks', 'Arakyd Vocabulator Model',
            'Axial Piston Model', 'Nanochip Model', 'AutoTerrain Tread Count', 'InfoCore Size', 'Sonoreceptors', 'Cranial Uplink Bandwidth']
parts_nominal = ['Arakyd Vocabulator Model',
                 'Axial Piston Model', 'Nanochip Model']
parts_nominal_data = {
  'Arakyd Vocabulator Model': ['model A','model B', 'model C','model D', 'model E'],
  'Axial Piston Model': ['alpha','beta', 'gamma','kappa','zeta',],
  'Nanochip Model': ['windows2000', 'v0.5', 'v1.0', 'v1.5', 'v2.0'],
}
parts_quant = ['Repulsorlift Motor HP', 'Astrogation Buffer Length', 'Polarity Sinks',
               'AutoTerrain Tread Count', 'InfoCore Size', 'Sonoreceptors', 'Cranial Uplink Bandwidth']

parts_coefficients = {
    'Repulsorlift Motor HP': 0,
    'Astrogation Buffer Length': 0,
    'Polarity Sinks': 0,
    'Arakyd Vocabulator Model': [],
    'Axial Piston Model': [],
    'Nanochip Model': [],
    'AutoTerrain Tread Count': 0,
    'InfoCore Size': 0,
    'Sonoreceptors': 0,
    'Cranial Uplink Bandwidth': 0
}
# Helper functions:
# initiating data structure, family: direct parents,children,siblings


def getParentChildrenSibling(id):
    family = [id]
    for i in genealogy.predecessors(id):
        family.append(i)
    for i in genealogy.successors(id):
        family.append(i)
    for parent in genealogy.predecessors(id):
        for sibling in genealogy.successors(parent):
            if id != sibling:
                family.append(sibling)
    return family


# start game
# game = game = rg.Robogame(KEY,server="127.0.0.1",port=5000)
game = rg.Robogame("VISION", server="roboviz.games")

game.setReady()
print("Set Ready")

# read initial data
robots = game.getRobotInfo()
tree = game.getTree()
genealogy = nx.tree_graph(tree)

# initiating data structure, family: direct parents,children,siblings
try:
  df = pd.read_pickle('df_temp.pkl')
except:
  df = robots[['id','expires']]
  df['family'] = df['id'].apply(lambda id: getParentChildrenSibling(id))
  df['dict'] = None
  df['locked'] = np.empty((len(df), 0)).tolist()
  df['infoCount'] = 0
  df['predicted_productivity'] = 0

  for part in partList:
      if(part in parts_nominal):
          df[part] = None
      else:
          df[part] = np.NaN
  df.set_index('id', drop=False, inplace=True)
  df.index.name = None


time_end = time.time()
print('Import time: ', time_end-time_start)

game.getGameTime()
print(game.getGameTime())

while(True):
   gametime = game.getGameTime()
   print(gametime)
   if ('Error' in gametime):
       if (gametime['Error'] != 'Waiting on other team'):
           print("Error"+str(gametime))
           break
       else:
           print("Wating on other team")
           time.sleep(1)
   else:
       timetogo = gametime['gamestarttime_secs'] - gametime['servertime_secs']
       if (timetogo <= 0):
           print("Let's go!")
           break
       print("waiting to launch... game will start in " + str(int(timetogo)))
       time.sleep(1) # sleep 1 second at a time, wait for the game to start


initial_bets = {}
for i in np.arange(0, 100):
    initial_bets[int(i)] = int(50)
game.setBets(initial_bets)


def updateDf(hints, robot_df):
    predictionHints = hints['predictions']
    partsHints = hints['parts']
    print('Prediction Hint length: ', len(predictionHints))
    print('Part Hint length: ', len(partsHints))
    for hint in predictionHints:
        id = hint['id']
        time = hint['time']
        value = hint['value']
        print(df.loc[id, 'locked'])
        df.loc[id, 'locked'].append(time)

        id_list = getParentChildrenSibling(id)
        id_list.append(id)
        for idx in id_list:
            temp = df.loc[idx, 'dict']
            if temp is None:
                df.loc[idx, 'dict'] = [{time: value}]
                df.loc[idx, 'infoCount'] = 1
                continue
            if time in temp.keys():
                locked = df.loc[idx, 'locked']
                if len(locked) == 0:
                    temp[time] = (temp[time] + value) / 2
                elif time in df.loc[idx, 'locked']:
                    continue
                else:
                    temp[time] = (temp[time] + value) / 2
            else:
                temp[time] = value
            df.loc[idx, 'dict'] = [temp]
            df.loc[idx, 'infoCount'] = len(temp)
    for hint in partsHints:
        id = hint['id']
        column = hint['column']
        value = hint['value']
        df.loc[id, column] = value
    for index, robot in robot_df.iterrows():
        id = index
        productivity = robot['Productivity']
        df.loc[id, 'productivity'] = productivity
        # print(robot)
    dfTmp = df[partList]
    dfTmp.fillna(0, inplace=True)
    dfTmp['predicted_productivity']=0
    for part in parts_quant:
        dfTmp['predicted_productivity'] += parts_coefficients[part]*dfTmp[part]
    for part in parts_nominal:
        dfTmp['predicted_productivity'] += dfTmp[part].apply(lambda x: 1 if x in parts_coefficients[part] else 0)
    df['predicted_productivity'] = dfTmp['predicted_productivity']
    df.to_pickle('df_temp.pkl')
    


def merge_dict(dict1, dict2):
    if not dict1 and not dict2:
        return dict()
    elif not dict1:
        return dict2
    elif not dict2:
        return dict1
    else:
        res = dict1
        for k, v in dict2.items():
            if k in dict1.keys():
                res[k] = (dict1[k]+v)/2
            else:
                res[k] = v
        return res


# ========================================================================== time series
def time_series_df_func(hints_df, robots):
  # hint_df : df in the datamodel.ipynb, robots: inital robots information
  ### transfer df => # ts_wth_fmly_df = time_series_df_func(hints_df) ###

    robots_100 = robots.sort_values(by=['expires'], ignore_index=True)[:100]
    robots = robots.sort_values(by=['expires'], ignore_index=True)
    merge_df = robots_100.merge(hints_df, how='left')
    time_list, value_list, original_list, id_list, family_list = [], [], [], [], []
    for index, row in robots_100.merge(hints_df, how='left').iterrows():
        i = row.dict
        if i is None:
            continue
        time_list += list(i.keys())
        value_list += list(i.values())
        original_list += list(i.values())
        id_list += [row.id]*len(i.keys())
        family_list += [row.family]*len(i.keys())
    time_series_df = pd.DataFrame(
        data=np.array([time_list, value_list,original_list, id_list, family_list]).T,
        columns=['time', 'value', 'original' ,'id', 'family']
    )
    time_list, value_list, if_original, id_list = [], [], [], []
    for row in list(df.family):
        num = 0
        for i in row:
            #             print(i,row,num)
            time_list += list(time_series_df[time_series_df.id == i].time)
            value = time_series_df[time_series_df.id == i].value
            value_list += list(value)
            if i == row[0]:
              if_original += list([True]*len(value))
            else:
              if_original += list([False]*len(value))
            num += len(value)
        id_list += [i]*num
    ts_wth_fmly_df = pd.DataFrame(
        data=np.array([time_list, value_list, if_original, id_list]).T,
        columns=['time', 'value', 'if_original', 'id']
    )
    return ts_wth_fmly_df


def timeseries_func(ts_df,robots, df=None, genealogy=None, current_time=None):
    ## ts_df: ts_wth_fmly_df = time_series_df_func(hints_df)
    robots_100 = robots.sort_values(by=['expires'],ignore_index=True)[:100]
    ts_basic = alt.Chart(ts_df).mark_circle(
        filled=True,size=90).encode(
        x='time:Q',
        y='value:Q',
        opacity=alt.condition(alt.datum.if_original == 1,alt.value(1),alt.value(0.2)),
        tooltip=['value:Q','id:N']
    ).properties(title="time vs value of all robots")
    
    if current_time == None: return ts_basic # original time series data
    else:
        selection=alt.selection_single(empty="none")

        timeslider=alt.binding_range(
            min=0,  # min
            max=100,  # max
            step=1,              # how many steps to move when the slider adjusts
            name="max_time"        # what we call this slider variable
            )
        timeslider1=alt.binding_range(
            min=0,  # min
            max=100,  # max
            step=1,              # how many steps to move when the slider adjusts
            name="min_time"        # what we call this slider variable
            )
        timeselector = alt.selection_single(
            bind=timeslider,        # bind to the slider
            fields= ["time"], # we'll use the cutoff variable
            init={"time":100}  # start at the max
            )
        timeselector1 = alt.selection_single(
            bind=timeslider1,        # bind to the slider
            fields= ["time"], # we'll use the cutoff variable
            init={"time":0}  # start at the max
            )
        ts_time_filter = ts_basic.add_selection(timeselector).add_selection(timeselector1).transform_filter(
                alt.datum.time < timeselector.time,
            ).transform_filter(
                alt.datum.time > timeselector1.time,
            )

        ts_time_filter_reg = ts_time_filter.transform_loess('time', 'value').mark_line()
        selectionInterval=alt.selection_interval()
        colorCondition=alt.condition(selectionInterval,alt.value("red"), alt.value("gray"))
        opacity_1 = alt.condition(selectionInterval,
                              alt.value(1), alt.value(0.6)
                              )

        ts_interval_filter =ts_time_filter.add_selection(
                                        selectionInterval
                                    ).encode(
                                    color = colorCondition)
        
        #======================================
        id_list = list(robots_100.id.unique())
        id_list.sort()
        dropdown = alt.binding_select(options=id_list,name="Select id: ")
        selection_id=alt.selection_single(
            fields=['id'], # our selection is going to select based on origin
            init={"id":1}, # what should the start value be? (Europe for us)
            # now bind the selection to the dropdown 
            bind=dropdown
        )
        selection_id = alt.selection_single(on='mouseover', empty='none', clear='mouseout', fields = ['id'])

        #step 2
        colorCondition_id = alt.condition(selection_id,alt.value("blue"),alt.value('lightgray'))


        ts_groupfilter = ts_basic.add_selection(selection_id).encode(
            color=colorCondition_id # step 4
        )
        ts_regression = ts_basic.add_selection(selection_id).transform_filter(
            selection_id).transform_regression('time', 'value',method = 'poly').mark_line().properties(width=300, height=300)
        

        id_infoCount = dict(zip(df.id, df.infoCount))
        id_expires = dict(zip(df.id, df.expires))
        id_predicted_productivity = dict(zip(df.id, df.predicted_productivity))
        r,h = 3,3
        G = genealogy
        for rank in range(0,h+1):
            nodes_in_rank = nx.descendants_at_distance(G, 0, rank)
            for node in nodes_in_rank: 
                G.nodes[node]['rank'] = rank
        pos = graphviz_layout(G, prog='dot') # neato
        pos_df = pd.DataFrame.from_records(dict(id=k, x=x, y=y) for k,(x,y) in pos.items())
        node_df = pd.DataFrame.from_records(dict(data, **{'id': n}) for n,data in G.nodes.data())
        node_df['rank'] = node_df['id'].apply(lambda x: id_infoCount[x])
        node_df['expires'] = node_df['id'].apply(lambda x: id_expires[x])
        node_df['predicted_productivity'] = node_df['id'].apply(lambda x: id_predicted_productivity[x])
        node_df['expires'] = node_df['expires'].fillna(0)
        node_df = node_df.rename(columns={'rank':'infoCount'})
        edge_data = ((dict(d, **{'edge_id':i, 'end':'source', 'id':s}),
                    dict(d, **{'edge_id':i, 'end':'target', 'id':t}))
                    for i,(s,t,d) in enumerate(G.edges.data()))
        edge_df = pd.DataFrame.from_records(chain.from_iterable(edge_data))

        x,y = alt.X('x:Q', axis=None), alt.Y('y:Q', axis=None)

        opacity_condition = alt.condition(selection_id, alt.value(1.0), alt.value(0))

        node_position_lookup = {
            'lookup': 'id', 
            'from_': alt.LookupData(data=pos_df, key='id', fields=['x', 'y'])
        }
        nodes = (
            alt.Chart(node_df)
            .mark_circle(size=200, opacity=1)
            .encode(x=x, y=y, tooltip=['id', 'infoCount', 'expires', 'predicted_productivity:Q'], color=alt.condition(alt.datum.expires>curr_time,alt.Color('predicted_productivity:Q', scale=alt.Scale(scheme="redblue")),alt.value('grey')))
            .transform_lookup(**node_position_lookup)
            .add_selection(selection_id)
        )
        highlight = (
            alt.Chart(node_df)
            .mark_circle(filled=False, color='red', strokeWidth=3, size=600)
            .encode(x=x, y=y, opacity=opacity_condition)
            # .add_selection(selection_id)
            .transform_filter(selection_id)
            .transform_lookup(**node_position_lookup)
        )
        edges = (
            alt.Chart(edge_df)
            .mark_line(color='gray')
            .encode(x=x, y=y, detail='edge_id:N')  # `detail` gives one line per edge
            .transform_lookup(**node_position_lookup)
        )
        chart = (
            (edges+nodes+highlight)
            .properties(width=1000, height=300)
            # .configure_view(strokeWidth=0)
        )

    return ts_regression | chart


def top_5_time_series_func(ts_df, ts_basic, robots, current_time=0):
    robots_100 = robots.sort_values(by=['expires'], ignore_index=True)[:100]
    top5_id = list(robots_100[robots_100.expires > current_time].id[:5])
    top5_time = list(robots_100[robots_100.expires > current_time].expires[:5])
    top1 = ts_basic.transform_filter(
        alt.datum.id == top5_id[0]
    ).properties(title="Top1: id="+str(top5_id[0]) + ", expire time= " + str(top5_time[0]))
    top1_smooth = top1.transform_regression(
        'time', 'value', method='poly').mark_line()
    line1 = alt.Chart(robots_100).mark_rule(color='red', size=2).encode(
        x='expires:Q').transform_filter(
        alt.datum.id == top5_id[0]
    )

    top2 = ts_basic.transform_filter(
        alt.datum.id == top5_id[1]
    ).properties(title="Top2: id="+str(top5_id[1]) + ", expire time= " + str(top5_time[1]))
    top2_smooth = top2.transform_regression(
        'time', 'value', method='poly').mark_line()
    line2 = alt.Chart(robots_100).mark_rule(color='red', size=2).encode(x='expires:Q').transform_filter(
        alt.datum.id == top5_id[1]
    )
    top3 = ts_basic.transform_filter(
        alt.datum.id == top5_id[2]
    ).properties(title="Top3: id="+str(top5_id[2]) + ", expire time= " + str(top5_time[2]))
    top3_smooth = top3.transform_regression(
        'time', 'value', method='poly').mark_line()
    line3 = alt.Chart(robots_100).mark_rule(color='red', size=2).encode(x='expires:Q').transform_filter(
        alt.datum.id == top5_id[2]
    )
    top4 = ts_basic.transform_filter(
        alt.datum.id == top5_id[3]
    ).properties(title="Top4: id="+str(top5_id[3]) + ", expire time= " + str(top5_time[3]))
    top4_smooth = top4.transform_regression(
        'time', 'value', method='poly').mark_line()
    line4 = alt.Chart(robots_100).mark_rule(color='red', size=2).encode(x='expires:Q').transform_filter(
        alt.datum.id == top5_id[3]
    )
    top5 = ts_basic.transform_filter(
        alt.datum.id == top5_id[4]
    ).properties(title="Top5: id="+str(top5_id[4]) + ", expire time= " + str(top5_time[4]))
    top5_smooth = top5.transform_regression(
        'time', 'value', method='poly').mark_line()
    line5 = alt.Chart(robots_100).mark_rule(color='red', size=2).encode(x='expires:Q').transform_filter(
        alt.datum.id == top5_id[4]
    )
    top1_all = (top1 + top1_smooth + line1).properties(
            width=180,
            height=200
        )
    top2_all = (top2 + top2_smooth + line2)
    top3_all = (top3 + top3_smooth + line3)
    top4_all = (top4 + top4_smooth + line4)
    top5_all = (top5 + top5_smooth + line5)

    ret = (top1_all | top2_all | top3_all | top4_all | top5_all).resolve_scale(x = 'shared', y = 'shared')
    return ret
# ========================================================================== time series


# ========================================================================== smallmultiple

def smallmultiple(df):
    # A preferable solution for this is to use the old dataframe serializer by setting this in your .streamlit/config.toml file:

    # [global]
    # dataFrameSerialization = "legacy"
    nominal_list = ['Arakyd Vocabulator Model',
                    'Axial Piston Model', 'Nanochip Model']
    # slider_productivity
    productivity_min = df['productivity'].min()
    productivity_max = df['productivity'].max()
    productivity_slider_max = alt.binding_range(
        min=productivity_min,
        max=productivity_max,
        step=0.01,
        name="productivity"
    )
    productivity_slider_min = alt.binding_range(
        min=productivity_min,
        max=productivity_max,
        step=0.01,
        name="productivity"
    )
    productivity_selector_max = alt.selection_single(
        bind=productivity_slider_max,
        fields=["productivity_max"],
        init={"productivity_max": productivity_max}
    )
    productivity_selector_min = alt.selection_single(
        bind=productivity_slider_min,
        fields=["productivity_min"],
        init={"productivity_min": productivity_min}
    )
    charts = []
    chartTmplate = alt.Chart(df).transform_filter(
        alt.datum['productivity'] < productivity_selector_max.productivity_max
    ).transform_filter(
        alt.datum['productivity'] > productivity_selector_min.productivity_min
    )
    for part in partList:
        if(part not in nominal_list):
            part = part+':Q'
            charts.append(
                chartTmplate.mark_point(filled=True,size=90).encode(
                    y=alt.Y('productivity:Q'),
                    x=alt.X(part, axis=alt.Axis(titleY=-215)),
                    tooltip=['id', 'productivity', part]
                ).properties(
                    width=200,
                    height=200
                ).add_selection(
                    productivity_selector_max,
                    productivity_selector_min
                )
            )
    # I don't know why, as long as I use a variable, the filter does not work. (for example use nominal_list[0] to replace 'Axial Piston Model')
    charts.append(
        chartTmplate.transform_filter(
            alt.datum['Axial Piston Model'] != "None"
        ).mark_point(filled=True,size=90).encode(
            y=alt.Y('productivity:Q'),
            x=alt.X('Axial Piston Model', axis=alt.Axis(titleY=-215)),
            tooltip=['id', 'productivity', 'Axial Piston Model']
        ).properties(
            width=200,
            height=200
        ).add_selection(
            productivity_selector_max,
            productivity_selector_min
        )
    )

    charts.append(
        chartTmplate.transform_filter(
            alt.datum['Arakyd Vocabulator Model'] != "None"
        ).mark_point(filled=True,size=90).encode(
            y=alt.Y('productivity:Q'),
            x=alt.X('Arakyd Vocabulator Model', axis=alt.Axis(titleY=-215)),
            tooltip=['id', 'productivity', 'Arakyd Vocabulator Model']
        ).properties(
            width=200,
            height=200
        ).add_selection(
            productivity_selector_max,
            productivity_selector_min
        )
    )

    charts.append(
        chartTmplate.transform_filter(
            alt.datum['Nanochip Model'] != "None"
        ).mark_point(filled=True,size=90).encode(
            y=alt.Y('productivity:Q'),
            x=alt.X('Nanochip Model', axis=alt.Axis(titleY=-215)),
            tooltip=['id', 'productivity', 'Nanochip Model']
        ).properties(
            width=200,
            height=200
        ).add_selection(
            productivity_selector_max,
            productivity_selector_min
        )
    )
    chart = alt.vconcat(*charts)
    line1 = charts[0:5]
    line2 = charts[5:10]
    chart = alt.vconcat(alt.hconcat(*line1), alt.hconcat(*line2)).configure_concat(
        spacing=20
    ).configure_axis(labelLimit=400)
    return chart

# ========================================================================== smallmultiple

# ========================================================================== boxplot
def boxChart(df):
    boxPlot = alt.Chart(df).transform_fold(
        ["productivity","predicted_productivity"]
    ).transform_filter(
      alt.datum.value != 0
    ).mark_boxplot(extent='min-max').encode(
        x= alt.X('key:O', title=""),
        y=alt.Y('value:Q', title="")
    ).properties(width=150, height=600)
    return boxPlot
#   ========================================================================== boxplot
# show select box. default is showing visualizations


for part in parts_quant:
  parts_coefficients[part] = st.sidebar.slider(part,-1.0, 1.0, 0.0)
for part in parts_nominal:
  parts_coefficients[part] = st.sidebar.multiselect(part,parts_nominal_data[part],[])


print('Game Preparation time: ', time.time() - time_end)
time_end = time.time()

#code for showing part and robot recommendations, and showing the entering interest page
parts_recommend = st.empty()
robots_recommend = st.empty()
input_robot_interest = []
input_parts_interest = []
if 'Entering Interests' in selectbox:
    input_parts_interest = st.multiselect(
        "Select one or more part interests:",
        partList)
    input_robot_interest = st.text_input(
        "Please enter robot interest, separate by a single white space:")
    try:
        input_robot_interest = [int(id) for id in input_robot_interest.split() if (
            int(id) <= 150) & (int(id) >= 0)]
    except:
        input_robot_interest = []


time_end = time.time()

# our main loop
for timeloop in np.arange(0, 100):
    mainloopStart = time.time()
    print(parts_coefficients)
    # update df, get info
    lasthints = {}
    gametime = game.getGameTime()
    curr_time = gametime['curtime']
    time_end = time.time()
    hints = game.getHints()
    print('getHint time: ', time.time() - time_end)
    time_end = time.time()
    robots = game.getRobotInfo()
    updateDf(hints, robots)


    print('update time: ', time.time() - time_end)
    time_end = time.time()




    # we change bets with new information
    bets = {}
    for id in range(100):
        expire_time = robots.iloc[id]['expires']
        if curr_time > expire_time:
            # ignore expired robots
            continue
        points = df.iloc[id]['dict']
        for family_member in df.iloc[id]['family']:
            points = merge_dict(points, df.iloc[id]['dict'])
        # now we have all the bot's family information and its own information, time to predict!
        # here we implement a basic strat, for robot A expiring at time ta, we gather its family information together into one chart, we can find tb<ta<tc, where tb and tc are the closest available family info we have on robot A, we bet on Va = (Vb-Vc)(ta-tc)/(tb-tc)+Vc
        ids = list(points.keys())
        if not ids:
            continue
        if expire_time <= min(ids) or expire_time >= max(ids):
            # cant predict yet, insufficient data
            continue
        ids += [expire_time]
        ids.sort()
        ind = ids.index(expire_time)
        ta = expire_time
        try:
            tb, tc = ids[ind-1], ids[ind+1]
        except:
            # print(ids, ind, expire_time)
            continue

        Vb, Vc = points[tb], points[tc]
        try:
            bets[id] = int((Vb-Vc)*(ta-tc)/(tb-tc)+Vc)
        except:
            continue
    # print("our bets", bets)
    game.setBets(bets)

    print('setbet time: ', time.time() - time_end)
    time_end = time.time()

    # here we set robot and parts interests
    # we set robots interests for the next 5 robots that will expire
    robot_interest = list(robots[robots['expires'] > curr_time].sort_values(
        by=['expires'], ascending=True).head(5)['id'])
    # we will randomly
    # as for parts interest, we prioritize on knowing about the last 5 robots
    last5_robots = list(robots[robots['expires'] < curr_time].sort_values(
        by=['expires'], ascending=False).head(5)['id'])
    # parts info count
    parts_dict = dict()
    for part in partList:
        parts_dict[part] = 0
    for id in last5_robots:
        for part in parts_nominal:
            if df.iloc[id][part] != '':
                parts_dict[part] += 1
        for part in parts_quant:
            if df.iloc[id][part] != np.NaN:
                parts_dict[part] += 1
    # we ask for info on the top 3 parts we have least information of
    parts_interest = heapq.nsmallest(3, parts_dict, key=parts_dict.get)

    # show page to let users enter interests
    if 'Entering Interests' in selectbox:
        parts_recommend.write('Recommended parts interest: '+",".join(parts_interest))
        robots_recommend.write('Recommended robot interest: ' +  ",".join([str(id) for id in robot_interest]))
        if len(input_parts_interest) > 0:
            parts_interest = input_parts_interest
        if len(input_robot_interest) > 0:
            robot_interest = input_robot_interest
    # show our visualizations
    if 'Visualization' in selectbox:

        current_game_time.markdown(f'## Current game time: {str(curr_time)}')
        time_end = time.time()
        robots_100 = robots.sort_values(by=['expires'], ignore_index=True)[:100]
        id_productivity = dict(zip(df.id, df.predicted_productivity))
        try:
            top10_id = list(robots_100[robots_100.expires > curr_time].id[0:9])
            top10_time = list(robots_100[robots_100.expires > curr_time].expires[0:9])
            table_time['id'] = top10_id
            table_time['expire'] = top10_time
            table_time['p'] = table_time['id'].apply(lambda x: id_productivity[x])
            table_time.index = pd.Series([1,2,3,4,5,6,7,8,9])
            table_t.write(table_time)
        except:
            pass
        # draw timeseries vis & tree vis
        time_end = time.time()
        ts_wth_fmly_df = time_series_df_func(df, robots)
        ts_basic = timeseries_func(ts_wth_fmly_df, robots)
        print('Generate dataframe and basic diagram time: ', time.time() - time_end)
        time_end = time.time()
        ts_top5_plot = top_5_time_series_func(ts_wth_fmly_df, ts_basic, robots, current_time=curr_time)
        print('top5 plot time: ', time.time() - time_end)
        time_end = time.time()
        
        ts_all_plot = timeseries_func(ts_wth_fmly_df, robots, df, genealogy, curr_time)
        print('tree plot time: ', time.time() - time_end)
        time_end = time.time()
        timevis1.write(ts_top5_plot)

        timevis2.write(ts_all_plot)
        

        print('timeseries+tree time: ', time.time() - time_end)
        time_end = time.time()


        # draw small multiple
        nonNadf = df[df['productivity'].notna()]
        if len(nonNadf) != 0:
            
            smallmultipleVis = smallmultiple(nonNadf)
            smvis.write(smallmultipleVis)
        
        
        print('smMultiple time: ', time.time() - time_end)
        time_end = time.time()

        # draw boxplot
        boxPlotDiplay = boxChart(df)
        boxPlot.write(boxPlotDiplay)

        # show recommended robot and parts interest
        parts_recommend.write('Recommended parts interest: '+",".join(parts_interest))
        robots_recommend.write('Recommended robot interest: ' +
                ",".join([str(id) for id in robot_interest]))

    # set robots and parts interest
    game.setPartInterest(parts_interest)
    game.setRobotInterest(robot_interest)

    print('setInterest time: ', time.time() - time_end)
    time_end = time.time()

    print("time: ", gametime['curtime'])
    # print("our bets", bets)
    print(robot_interest)
    print(parts_interest)

    # sleep to 6
    mainloopEnd = time.time()
    if(mainloopEnd - mainloopStart < 6):
      time.sleep(6 - (mainloopEnd - mainloopStart))



