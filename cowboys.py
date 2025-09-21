import streamlit as st
import sqlite3
import pandas as pd
from streamlit_js_eval import streamlit_js_eval
import hashlib
from datetime import datetime as dt
from datetime import timedelta as td
import warnings
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype)
from github import Github
import requests
import base64
import subprocess
from streamlit import session_state as ss
import os
import numpy as np
from streamlit_modal import Modal
import st_cookie

version = '1.1'
counter=0
warnings.filterwarnings("ignore")
st.set_page_config(layout='wide',initial_sidebar_state='expanded')
st.markdown("""
                <html>
                <style>
                        ::-webkit-scrollbar {
                            width: 2vw;
                            }

                            /* Track */
                            ::-webkit-scrollbar-track {
                            background: #f1f1f1;
                            }

                            /* Handle */
                            ::-webkit-scrollbar-thumb {
                            background: #888;
                            }

                            /* Handle on hover */
                            ::-webkit-scrollbar-thumb:hover {
                            background: #555;
                            }
                </style>
            """, unsafe_allow_html=True)

def filter_dataframe(df: pd.DataFrame, coll=[]) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    global counter
    counter += 1
    modify = st.checkbox("Add filters",value=True,key='checkbox' + str(counter))

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() <= 20 or column in coll:
                if df[column].nunique() <= 20:
                    user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()))
                else:
                    user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=[])
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def updatedb(sql):
    try:
        subprocess.check_output(['cp', 'cowboys.db', 'cowboysgh.db'])
        connection = sqlite3.connect('cowboys.db')
        cursor = connection.cursor()
        if sql.find(';') > 0:
            sqllist = sql.split(';')
            for i in sqllist:
                if len(i) > 1:
                    cursor.execute(i)
        else:            
            cursor.execute(sql)
        dblist = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'",connection)['name'].tolist()
        if not 'newbuyers' in dblist:
            cursor.execute('CREATE TABLE newbuyers AS SELECT * FROM buyers WHERE 1 = 0')
            cursor.execute('ALTER TABLE newbuyers ADD Contact TEXT')
        connection.commit()
        connection.close()
        token = st.secrets['TOKEN']
        repopath = 'skotrla/cowboys'
        filename = 'cowboys.db'
        g = Github(token)      
        repo = g.get_repo(repopath)
        headers = {"content-type": "application/json",
                   "authorization": f"token {token}",
                   "accept": "application/vnd.github+json"}
        sha = str(subprocess.check_output(['git', 'hash-object', 'cowboysgh.db']))[2:-3]
        with open(filename, "rb") as source_file:
            encoded_string = base64.b64encode(source_file.read()).decode("utf-8")
        payload = {"message": f"Uploaded file at {dt.utcnow().isoformat()}",
                   "content": encoded_string,
                   "sha": sha}
        r = requests.put(f'https://api.github.com/repos/{repopath}/contents/{filename}',json=payload,headers=headers)
        if str(r).find('200') >= 0:
            subprocess.check_output(['rm', 'cowboysgh.db'])
        else:
            if r.text.find(sha) >= 0:
                contents = repo.get_contents(filename)
                subprocess.check_output(['rm', 'cowboysgh.db'])
                with open('cowboysgh.db', 'wb') as binary_file:
                    binary_file.write(base64.b64decode(contents.content))
        if sql.find('newbuyers') >= 0:
            modal = Modal(key="OK",title="Listing Added")
            modal.open()
            with modal.container():
                st.write('New User Listings Will Not Show Up Until Approved, You Will Be Sent a Login via Contact Info')
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
    except Exception as e:
        st.write(getattr(e, 'message', str(e)))
        st.write(sql)

def stlog(logstr):
    os.write(1,(logstr+'\n').encode())

def menu(page):
    html = f'<a href="https://www.facebook.com/groups/1041799047402614">Facebook Group</a>'
#    if st.session_state.auth != 'Y':
#        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=buyers">Buyers</a>'
#        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=sellers">Sellers</a>'
#        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=login">Login</a>'
#    else:
        #html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=buyers">Buyers</a>'
        #html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=sellers">Sellers</a>'
#        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=buyers&user={st.session_state.user}&hash={st.session_state.hash}">Buyers</a>'
#        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=sellers&user={st.session_state.user}&hash={st.session_state.hash}">Sellers</a>'
#        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=logout">Welcome {st.session_state.user}- Click to Logout</a>'                   
    if st.session_state.auth != 'Y':
        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=users">VETTED Sellers</a>'
        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=buyers">Buyers</a>'
        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=sellers">Sellers</a>'
        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=login">Login</a>'
    else:
        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=users&user={st.session_state.user}&hash={st.session_state.hash}">VETTED Sellers</a>'
        if page == 'buyers':
            html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=buyers&a=N&user={st.session_state.user}&hash={st.session_state.hash}">All Buyers</a>'
        else:
            html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=buyers&user={st.session_state.user}&hash={st.session_state.hash}">Buyers</a>'
        if page == 'sellers':
            html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=sellers&a=N&user={st.session_state.user}&hash={st.session_state.hash}">All Sellers</a>'
        else:
            html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=sellers&user={st.session_state.user}&hash={st.session_state.hash}">Sellers</a>'
        html += f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a target="_self" href="?page=logout">Welcome {st.session_state.user}- Click to Logout</a>'                   
    return html    

def showpage(page):
    match page:
        case 'games':
            connection = sqlite3.connect('cowboys.db')
            cursor = connection.cursor()
            games = pd.read_sql(f'SELECT Game FROM games',connection)
            connection.close()
            st.markdown(menu('games'),unsafe_allow_html=True)
            st.title('2025 Dallas Cowboys Games')
            st.data_editor(games,hide_index=True)
    #        if user[0] == 'scott.kotrla' and hash[0] == hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest():
            if st.session_state.user == 'scott.kotrla' and st.session_state.auth == 'Y':
                form=st.form(key='games')
                with form:
                    name = st.text_input("Game")
                    submit = st.form_submit_button("Submit")
                    if submit:        
                        updatedb(f'INSERT INTO games (Game) VALUES ("{name}")')
        case 'areas':
            connection = sqlite3.connect('cowboys.db')
            cursor = connection.cursor()
            areas = pd.read_sql(f'SELECT Area,Description FROM areas',connection)
            connection.close()
            st.markdown(menu('areas'),unsafe_allow_html=True)
            st.title('Dallas Cowboys Stadium Areas')
            st.data_editor(areas,hide_index=True)
    #        if user[0] == 'scott.kotrla' and hash[0] == hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest():
            if st.session_state.user == 'scott.kotrla' and st.session_state.auth == 'Y':
                form=st.form(key='areas')
                with form:
                    name = st.text_input("Area")
                    description = st.text_input("Description")
                    submit = st.form_submit_button("Submit")
                    if submit:        
                        updatedb(f'INSERT INTO areas (Area,Description) VALUES ("{name}","{description}")')
        case 'users':
    #        if user[0] == 'scott.kotrla' and hash[0] == hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest():
            if st.session_state.user == 'scott.kotrla' and st.session_state.auth == 'Y':
                connection = sqlite3.connect('cowboys.db')
                cursor = connection.cursor()
                users = pd.read_sql(f'SELECT Name,Contact,Seller FROM users',connection)
                connection.close()
                st.markdown(menu('users'),unsafe_allow_html=True)
                st.title('Users')
                st.data_editor(users,hide_index=True)
                form=st.form(key='users')
                with form:
                    name = st.text_input("Name")
                    contact = st.text_input("Contact")
                    seller = st.text_input("Seller")
                    submit = st.form_submit_button("Submit")
                    if submit:        
                        updatedb(f'INSERT INTO users (Name,Contact,Seller) VALUES ("{name}","{contact}","{seller}")')
            else:
                connection = sqlite3.connect('cowboys.db')
                cursor = connection.cursor()
                users = pd.read_sql(f'SELECT Name,Contact FROM users WHERE Seller="Y" ORDER BY Name',connection)
                connection.close()
                c2 = st.container()
                c2.markdown(menu('users'),unsafe_allow_html=True)
                c2.title('VETTED Sellers')
                c2.dataframe(filter_dataframe(users,users.columns.tolist()),hide_index=True, column_config={'Contact':st.column_config.LinkColumn()})
        case 'sellers':
            connection = sqlite3.connect('cowboys.db')
            cursor = connection.cursor()
    #        seller = pd.read_sql(f'''SELECT * from users WHERE Contact = "{'https://m.me/' + user[0]}"''',connection)
            seller = pd.read_sql(f'''SELECT * from users WHERE Contact = "{'https://m.me/' + st.session_state.user}"''',connection)
            if len(seller) == 1:
                contact = seller['Contact'].tolist()[0]
                s = seller['Seller'].tolist()[0]
                seller = seller['Name'].tolist()[0]
            else:
                s = 'N'

            games = pd.read_sql(f'SELECT * FROM games', connection)
            games.insert(1,'Date', games['Game'].str.find('/'))
            fdate = str(dt.now()).replace(str(dt.now().year),str(dt.now().year+1))[:10] 
            games['Date'] = np.where(games['Date'] > 0,games['Game'].apply(lambda x: x[x.find('/')-2:]),fdate)
            games['Date'] = pd.to_datetime(games['Date'],format='mixed')
            gamelist = games[games['Date']>=dt.now()-td(days=1)]['Game'].tolist()
            #gamelist = pd.read_sql(f'SELECT * FROM games', connection)['Game'].tolist()
            arealist = pd.read_sql(f'SELECT * FROM areas', connection)['Area'].tolist()
            areas = pd.read_sql(f'SELECT Area,Description as Sections FROM areas',connection)        
    #        if hash[0] != hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest() or s != 'Y':
            if st.session_state.auth != 'Y' or s != 'Y' or a[0] == 'N':
                sql1 = f'SELECT Game, Area, Seller, Max(Last_Update) as Last_Update FROM sellers GROUP BY Game, Area, Seller'
                sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Low_Price(ea)", t2."High_Price(ea)", t2.Parking_Included, t2.Details, t1.Seller, t3.Contact, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN sellers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Seller=t2.Seller LEFT JOIN users t3 ON t2.Seller=t3.Name WHERE t2.Min_Qty > 0 ORDER BY t2."Low_Price(ea)"'
                sellers = pd.read_sql(sql2,connection)
                connection.close()
                sellers.insert(1,'Date', sellers['Game'].str.find('/'))
                fdate = str(dt.now()).replace(str(dt.now().year),str(dt.now().year+1))[:10] 
                sellers['Date'] = np.where(sellers['Date'] > 0,sellers['Game'].apply(lambda x: x[x.find('/')-2:]),fdate)
                sellers['Date'] = pd.to_datetime(sellers['Date'],format='mixed')
                sellers = sellers[sellers['Date']>=dt.now()-td(days=1)]
                sellers['Date'] = sellers['Date'].astype('str').str[:10]
                sellers = sellers.sort_values(['Date','Low_Price(ea)'])
                with st.sidebar.expander('Areas'):
                    c1 = st.container()
                    c1.dataframe(areas,hide_index=True)
                c2 = st.container()
                c2.markdown(menu('sellers'),unsafe_allow_html=True)
                c2.title('Sellers')
                c2.dataframe(filter_dataframe(sellers,sellers.columns.tolist()),hide_index=True, column_config={'Low_Price(ea)':st.column_config.NumberColumn(label='Low Price (ea)', format='$%d'),
                                                                     'High_Price(ea)':st.column_config.NumberColumn(label='High Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?'),
                                                                     'Contact':st.column_config.LinkColumn()})
            else:
                sql1 = f'SELECT Game, Area, Seller, Max(Last_Update) as Last_Update FROM sellers GROUP BY Game, Area, Seller'
    #            sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Low_Price(ea)", t2."High_Price(ea)", t2.Parking_Included, t2.Details, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN sellers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Seller=t2.Seller LEFT JOIN users t3 ON t2.Seller=t3.Name WHERE SUBSTR(t3.Contact,14,100) = "{user[0]}" AND t2.Min_Qty > 0 ORDER BY t2."Low_Price(ea)"'
                sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Low_Price(ea)", t2."High_Price(ea)", t2.Parking_Included, t2.Details, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN sellers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Seller=t2.Seller LEFT JOIN users t3 ON t2.Seller=t3.Name WHERE SUBSTR(t3.Contact,14,100) = "{st.session_state.user}" AND t2.Min_Qty > 0 ORDER BY t2."Low_Price(ea)"'
                sellers = pd.read_sql(sql2,connection)
                connection.close()
                sellers.insert(0,'Selected',False)
                with st.sidebar.expander('Areas'):
                    c1 = st.container()
                    c1.dataframe(areas,hide_index=True)
                c2 = st.container()
                c2.markdown(menu('sellers'),unsafe_allow_html=True)
                c2.title(f'Seller = {seller}')
                if len(sellers) > 0:
                    ss.edited_df = c2.data_editor(filter_dataframe(sellers,sellers.columns.tolist()),hide_index=True, column_config={'Low_Price(ea)':st.column_config.NumberColumn(label='Low Price (ea)', format='$%d'),
                                                                     'High_Price(ea)':st.column_config.NumberColumn(label='High Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?')})
                    formd=c2.form(key='delete')
                    with formd:
                        submit = st.form_submit_button("Delete Selected Rows")
                        if submit:
                            sql = ''
                            last_update = str(dt.now())[:19]
                            for index, row in ss.edited_df[ss.edited_df['Selected']==True].iterrows():
                                game = row['Game']
                                area = row['Area']
                                min_qty = 0
                                mq = 0
                                low_price = row['Low_Price(ea)']
                                hp = row['High_Price(ea)']
                                parking_included = row['Parking_Included']
                                details = row['Details']
                                sql += f'INSERT INTO sellers (Game, Area, Min_Qty, Max_Qty, "Low_Price(ea)", "High_Price(ea)", Parking_Included, Details, Seller, Last_Update) VALUES ("{game}","{area}",{min_qty},{mq},{low_price},{hp},"{parking_included[0]}","{details}","{seller}","{last_update}");'
                            if len(sql) > 1:
                                updatedb(sql)
                            else:
                                st.write('Nothing selected')
                    formc=c2.form(key='change')
                    with formc:
                        submit = st.form_submit_button("Update Changed Rows")
                        if submit:
                            sql = ''
                            last_update = str(dt.now())[:19]
                            for index, row in ss.edited_df.iterrows():
                                game = row['Game']
                                area = row['Area']
                                srow = sellers[(sellers['Game']==game) & (sellers['Area']==area)]
                                update = False
                                if len(srow) == 1:
                                    min_qty = row['Min_Qty']
                                    mq = row['Max_Qty']
                                    low_price = row['Low_Price(ea)']
                                    hp = row['High_Price(ea)']
                                    parking_included = row['Parking_Included']
                                    details = row['Details']
                                    if min_qty != srow['Min_Qty'].tolist()[0]:
                                        update = True
                                        if min_qty is None:
                                            min_qty = 0
                                    if mq != srow['Max_Qty'].tolist()[0]:
                                        update = True
                                        if mq is None:
                                            mq = min_qty
                                        else:
                                            if mq < min_qty:
                                                mq = min_qty
                                    if low_price != srow['Low_Price(ea)'].tolist()[0]:
                                        update = True
                                        if low_price is None:
                                            low_price = 0
                                    if hp != srow['High_Price(ea)'].tolist()[0]:
                                        update = True
                                        if hp is None:
                                            hp = low_price
                                        else:
                                             if hp < low_price:
                                                hp = low_price         
                                    if parking_included != srow['Parking_Included'].tolist()[0]:
                                        update = True
                                        if parking_included != 'Y':
                                            parking_included = 'N'
                                    if details != srow['Details'].tolist()[0]:
                                        update = True
                                        if len(details) == 0:
                                            details = ' '
                                    if update and (min_qty == 0 or low_price > 0):
                                        sql += f'INSERT INTO sellers (Game, Area, Min_Qty, Max_Qty, "Low_Price(ea)", "High_Price(ea)", Parking_Included, Details, Seller, Last_Update) VALUES ("{game}","{area}",{min_qty},{mq},{low_price},{hp},"{parking_included[0]}","{details}","{seller}","{last_update}");'
                                else:
                                    if len(srow) > 1:
                                        st.write('Duplicate rows found')
                                    if len(srow) == 0:
                                        st.write('No rows found')                            
                            if len(sql) > 1:
                                updatedb(sql)
                            else:
                                st.write('No changes')
                else:
                    ss.edited_df = c2.dataframe(filter_dataframe(sellers,sellers.columns.tolist()),hide_index=True, column_config={'Low_Price(ea)':st.column_config.NumberColumn(label='Low Price (ea)', format='$%d'),
                                                                     'High_Price(ea)':st.column_config.NumberColumn(label='High Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?')})                
                form=st.sidebar.form(key='sellers')
                with form:
                    st.title('Add Seller Listing')
                    game = st.selectbox("Game",gamelist)
                    area = st.selectbox("Area",arealist)
                    min_qty = st.number_input("Min Qty",min_value=0,value=0,step=1)
                    max_qty = st.number_input("Max Qty",min_value=0,value=0,step=1)
                    low_price = st.number_input("Low Price (ea)",min_value=0, value=0,step=1)
                    high_price = st.number_input("High Price (ea)",min_value=0, value=0,step=1)
                    parking_included = st.selectbox("Parking Included?",['N','Y'])
                    details = st.text_input("Details",value='')
                    submit = st.form_submit_button("Add Seller Listing")
                    if submit:        
                        if high_price < low_price:
                            hp = low_price
                        else:
                            hp = high_price
                        if max_qty < min_qty:
                            mq = min_qty
                        else:
                            mq = max_qty
                        if len(details) > 0 and (min_qty == 0 or low_price > 0):
                            last_update = str(dt.now())[:19]
                            updatedb(f'INSERT INTO sellers (Game, Area, Min_Qty, Max_Qty, "Low_Price(ea)", "High_Price(ea)", Parking_Included, Details, Seller, Last_Update) VALUES ("{game}","{area}",{min_qty},{mq},{low_price},{hp},"{parking_included[0]}","{details}","{seller}","{last_update}")')
                        else:
                            if len(details) > 0:
                                st.sidebar.write('Price must be > $0')
                            else:
                                st.sidebar.write('Details must not be blank')                            
        case 'buyers':
            connection = sqlite3.connect('cowboys.db')
            cursor = connection.cursor()
    #        buyer = pd.read_sql(f'''SELECT * from users WHERE Contact = "{'https://m.me/' + user[0]}"''',connection)
            buyer = pd.read_sql(f'''SELECT * from users WHERE Contact = "{'https://m.me/' + st.session_state.user}"''',connection)
            if len(buyer) == 1:
                contact = buyer['Contact'].tolist()[0]
                buyer = buyer['Name'].tolist()[0]
            gamelist = pd.read_sql(f'SELECT * FROM games', connection)['Game'].tolist()
            arealist = pd.read_sql(f'SELECT * FROM areas', connection)['Area'].tolist()
            areas = pd.read_sql(f'SELECT Area,Description as Sections FROM areas',connection)        
    #        if hash[0] != hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest():
            if st.session_state.auth != 'Y' or a[0] == 'N':
                sql1 = f'SELECT Game, Area, Buyer, Max(Last_Update) as Last_Update FROM buyers GROUP BY Game, Area, Buyer'
                sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Price(ea)", t2.Parking_Included, t2.Details, t1.Buyer, t3.Contact, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN buyers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Buyer=t2.Buyer LEFT JOIN users t3 ON t2.Buyer=t3.Name WHERE t2.Min_Qty > 0 ORDER BY t2."Price(ea)" DESC'
                buyers = pd.read_sql(sql2,connection)
                if 'name' in st.session_state:
                    sql1 = f'SELECT Game, Area, Buyer, Contact, Max(Last_Update) as Last_Update FROM newbuyers GROUP BY Game, Area, Buyer, Contact'
                    sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Price(ea)", t2.Parking_Included, t2.Details, t1.Buyer, t1.Contact, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN newbuyers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Buyer=t2.Buyer AND t1.Contact=t2.Contact WHERE t2.Min_Qty > 0 AND t2.Buyer="{st.session_state.name}" AND t2.Contact="{st.session_state.contact}"'
                    newbuyers = pd.read_sql(sql2,connection)
                connection.close()
                buyers.insert(1,'Date', buyers['Game'].str.find('/'))
                fdate = str(dt.now()).replace(str(dt.now().year),str(dt.now().year+1))[:10] 
                buyers['Date'] = np.where(buyers['Date'] > 0,buyers['Game'].apply(lambda x: x[x.find('/')-2:]),fdate)
                buyers['Date'] = pd.to_datetime(buyers['Date'],format='mixed')
                buyers = buyers[buyers['Date']>=dt.now()]
                buyers['Date'] = buyers['Date'].astype('str').str[:10]
                buyers = buyers.sort_values(['Date','Price(ea)'])
                with st.sidebar.expander('Areas'):
                    c1 = st.container()
                    c1.dataframe(areas,hide_index=True)
                c2 = st.container()
                c2.markdown(menu('buyers'),unsafe_allow_html=True)
                c2.title('Buyers')
                c2.dataframe(filter_dataframe(buyers,buyers.columns.tolist()),hide_index=True, column_config={'Price(ea)':st.column_config.NumberColumn(label='Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?'),
                                                                     'Contact':st.column_config.LinkColumn()})                
                if 'name' in st.session_state:
                    c3 = st.container()
                    c3.title('Pending Listings for Current User')
                    c3.dataframe(filter_dataframe(newbuyers,newbuyers.columns.tolist()),hide_index=True)
                else:
                    form=st.sidebar.form(key='buyers')
                    with form:
                        st.title('Add Buyer Listing - New User Listings Will Not Show Up Until Approved, You Will Be Sent a Login via Contact Info') 
                        game = st.selectbox("Game",gamelist)
                        area = st.selectbox("Area",arealist)
                        min_qty = st.number_input("Min Qty",min_value=0,value=0,step=1)
                        max_qty = st.number_input("Max Qty",min_value=0,value=0,step=1)
                        price = st.number_input("Price (ea)",min_value=0, value=0,step=1)
                        parking_included = st.selectbox("Parking Included?",['N','Y'])
                        details = st.text_input("Details",value='')
                        name = st.text_input("Full Name",value='')
                        contact  = st.text_input("Contact Info",value='')
                        submit = st.form_submit_button("Add Buyer Listing")
                        if submit:        
                            if max_qty < min_qty:
                                mq = min_qty
                            else:
                                mq = max_qty
                            if len(details) > 0 and (min_qty == 0 or price > 0) and len(name) > 0 and len(contact) > 0:
                                st.session_state.name = name
                                st.session_state.contact = contact                        
                                last_update = str(dt.now())[:19]
                                updatedb(f'INSERT INTO newbuyers (Game, Area, Min_Qty, Max_Qty, "Price(ea)", Parking_Included, Details, Buyer, Last_Update, Contact) VALUES ("{game}","{area}",{min_qty},{mq},{price},"{parking_included[0]}","{details}","{name}","{last_update}","{contact}")')
                            else:
                                if price <= 0:
                                    st.sidebar.write('Price must be > $0')
                                if len(details) <= 0:
                                    st.sidebar.write('Details must not be blank')                            
                                if len(name) < 8:
                                    st.sidebar.write('Full Name must be at least 8 characters')                            
                                if len(contact) < 8:
                                    st.sidebar.write('Contact Info must be at least 8 characters')                            
            else:
                sql1 = f'SELECT Game, Area, Buyer, Max(Last_Update) as Last_Update FROM buyers GROUP BY Game, Area, Buyer'
                sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Price(ea)", t2.Parking_Included, t2.Details, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN buyers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Buyer=t2.Buyer LEFT JOIN users t3 ON t2.Buyer=t3.Name WHERE SUBSTR(t3.Contact,14,100) = "{st.session_state.user}" AND t2.Min_Qty > 0 ORDER BY t2."Price(ea)" DESC'
                buyers = pd.read_sql(sql2,connection)
                connection.close()
                buyers.insert(0,'Selected',False)
                with st.sidebar.expander('Areas'):
                    c1 = st.container()
                    c1.dataframe(areas,hide_index=True)
                c2 = st.container()
                c2.markdown(menu('buyers'),unsafe_allow_html=True)
                c2.title(f'Buyer = {buyer}')
                if len(buyers) > 0:
                    ss.edited_df = c2.data_editor(filter_dataframe(buyers,buyers.columns.tolist()),hide_index=True, column_config={'Price(ea)':st.column_config.NumberColumn(label='Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?')})
                    formd=c2.form(key='delete')
                    with formd:
                        submit = st.form_submit_button("Delete Selected Rows")
                        if submit:
                            sql = ''
                            last_update = str(dt.now())[:19]
                            for index, row in ss.edited_df[ss.edited_df['Selected']==True].iterrows():
                                game = row['Game']
                                area = row['Area']
                                min_qty = 0
                                mq = 0
                                price = row['Price(ea)']
                                parking_included = row['Parking_Included']
                                details = row['Details']
                                sql += f'INSERT INTO buyers (Game, Area, Min_Qty, Max_Qty, "Price(ea)", Parking_Included, Details, Buyer, Last_Update) VALUES ("{game}","{area}",{min_qty},{mq},{price},"{parking_included[0]}","{details}","{buyer}","{last_update}");'
                            if len(sql) > 1:
                                updatedb(sql)
                            else:
                                st.write('Nothing selected')
                    formc=c2.form(key='change')
                    with formc:
                        submit = st.form_submit_button("Update Changed Rows")
                        if submit:
                            sql = ''
                            last_update = str(dt.now())[:19]
                            for index, row in ss.edited_df.iterrows():
                                game = row['Game']
                                area = row['Area']
                                srow = buyers[(buyers['Game']==game) & (buyers['Area']==area)]
                                update = False
                                if len(srow) == 1:
                                    min_qty = row['Min_Qty']
                                    mq = row['Max_Qty']
                                    price = row['Price(ea)']
                                    parking_included = row['Parking_Included']
                                    details = row['Details']
                                    if min_qty != srow['Min_Qty'].tolist()[0]:
                                        update = True
                                        if min_qty is None:
                                            min_qty = 0
                                    if mq != srow['Max_Qty'].tolist()[0]:
                                        update = True
                                        if mq is None:
                                            mq = min_qty
                                        else:
                                            if mq < min_qty:
                                                mq = min_qty
                                    if price != srow['Price(ea)'].tolist()[0]:
                                        update = True
                                        if price is None:
                                            price = 0
                                    if parking_included != srow['Parking_Included'].tolist()[0]:
                                        update = True
                                        if parking_included != 'Y':
                                            parking_included = 'N'
                                    if details != srow['Details'].tolist()[0]:
                                        update = True
                                        if len(details) == 0:
                                            details = ' '
                                    if update and (min_qty == 0 or price > 0):
                                        sql += f'INSERT INTO buyers (Game, Area, Min_Qty, Max_Qty, "Price(ea)", Parking_Included, Details, Buyer, Last_Update) VALUES ("{game}","{area}",{min_qty},{mq},{price},"{parking_included[0]}","{details}","{buyer}","{last_update}");'
                                else:
                                    if len(srow) > 1:
                                        st.write('Duplicate rows found')
                                    if len(srow) == 0:
                                        st.write('No rows found')                            
                            if len(sql) > 1:
                                updatedb(sql)
                            else:
                                st.write('No changes')
                else:
                    ss.edited_df = c2.dataframe(filter_dataframe(buyers,buyers.columns.tolist()),hide_index=True, column_config={'Price(ea)':st.column_config.NumberColumn(label='Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?')})                                
                form=st.sidebar.form(key='buyers')
                with form:
                    st.title('Add Buyer Listing')
                    game = st.selectbox("Game",gamelist)
                    area = st.selectbox("Area",arealist)
                    min_qty = st.number_input("Min Qty",min_value=0,value=0,step=1)
                    max_qty = st.number_input("Max Qty",min_value=0,value=0,step=1)
                    price = st.number_input("Price (ea)",min_value=0, value=0,step=1)
                    parking_included = st.selectbox("Parking Included?",['N','Y'])
                    details = st.text_input("Details",value='')
                    submit = st.form_submit_button("Add Buyer Listing")
                    if submit:        
                        if max_qty < min_qty:
                            mq = min_qty
                        else:
                            mq = max_qty
                        if len(details) > 0 and (min_qty == 0 or price > 0):
                            last_update = str(dt.now())[:19]
                            updatedb(f'INSERT INTO buyers (Game, Area, Min_Qty, Max_Qty, "Price(ea)", Parking_Included, Details, Buyer, Last_Update) VALUES ("{game}","{area}",{min_qty},{mq},{price},"{parking_included[0]}","{details}","{buyer}","{last_update}")')
                        else:
                            if len(details) > 0:
                                st.sidebar.write('Price must be > $0')
                            else:
                                st.sidebar.write('Details must not be blank')                            
        case 'newbuyers':
            connection = sqlite3.connect('cowboys.db')
            cursor = connection.cursor()
            gamelist = pd.read_sql(f'SELECT * FROM games', connection)['Game'].tolist()
            arealist = pd.read_sql(f'SELECT * FROM areas', connection)['Area'].tolist()
            areas = pd.read_sql(f'SELECT Area,Description as Sections FROM areas',connection)        
    #        if user[0] == 'scott.kotrla' and hash[0] == hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest():
            if st.session_state.user == 'scott.kotrla' and st.session_state.auth == 'Y':
                sql1 = f'SELECT Game, Area, Buyer, Contact, Max(Last_Update) as Last_Update FROM newbuyers GROUP BY Game, Area, Buyer, Contact'
                sql2 = f'SELECT t1.Game, t1. Area, t2.Min_Qty, t2.Max_Qty, t2."Price(ea)", t2.Parking_Included, t2.Details, t1.Buyer, t1.Contact, t1.Last_Update FROM ({sql1}) t1 LEFT JOIN newbuyers t2 ON t1.Last_Update=t2.Last_Update AND t1.Game=t2.Game AND t1.Area=t2.Area AND t1.Buyer=t2.Buyer AND t1.Contact=t2.Contact WHERE t2.Min_Qty > 0'
                newbuyers = pd.read_sql(sql2,connection)
                connection.close()
                newbuyers.insert(0,'Selected',False)
                with st.sidebar.expander('Areas'):
                    c1 = st.container()
                    c1.dataframe(areas,hide_index=True)
                c2 = st.container()
                c2.markdown(menu('newbuyers'),unsafe_allow_html=True)
                c2.title(f'New Buyers')
                if len(newbuyers) > 0:
                    ss.edited_df = c2.data_editor(filter_dataframe(newbuyers,newbuyers.columns.tolist()),hide_index=True, column_config={'Price(ea)':st.column_config.NumberColumn(label='Price (ea)', format='$%d'),
                                                                     'Max_Qty':st.column_config.NumberColumn(label='Max Qty', format='%d'),
                                                                     'Min_Qty':st.column_config.NumberColumn(label='Min Qty', format='%d'),
                                                                     'Parking_Included':st.column_config.TextColumn(label='Parking Included?')})
                    formd=c2.form(key='delete')
                    with formd:
                        submit = st.form_submit_button("Delete Selected Rows")
                        if submit:
                            sql = ''
                            last_update = str(dt.now())[:19]
                            for index, row in ss.edited_df[ss.edited_df['Selected']==True].iterrows():
                                game = row['Game']
                                area = row['Area']
                                min_qty = 0
                                mq = 0
                                price = row['Price(ea)']
                                parking_included = row['Parking_Included']
                                details = row['Details']
                                buyer = row['Buyer']
                                contact = row['Contact']
                                sql += f'INSERT INTO newbuyers (Game, Area, Min_Qty, Max_Qty, "Price(ea)", Parking_Included, Details, Buyer, Last_Update, Contact) VALUES ("{game}","{area}",{min_qty},{mq},{price},"{parking_included[0]}","{details}","{buyer}","{last_update}","{contact}");'
                            if len(sql) > 1:
                                updatedb(sql)
                            else:
                                st.write('Nothing selected')
        case 'logout':
            st.session_state.user = ''
            st.session_state.hash = ''
            st.session_state.auth = 'N'
            setcookie = f'const d = new Date(); d.setTime(d.getTime() + (365 * 24 * 60 * 60 * 1000)); const expires = "expires=" + d.toUTCString(); document.cookie = "cowboys={st.session_state.user + '&' + st.session_state.hash};" + expires + ";path=/";'
            streamlit_js_eval(js_expressions=setcookie)
            showpage('sellers')
        case 'login':
            reload = False
            form=st.form(key='login')
            with form:
                user = st.text_input("User ID")
                hash = st.text_input("Password")
                submit = st.form_submit_button("Submit")
                if submit:
                    if hash.strip() == hashlib.sha256((user.strip()+st.secrets['MYKEY']).encode()).hexdigest():
                        st.session_state.user = user
                        st.session_state.hash = hash
                        st.session_state.auth = 'Y'
                        #setcookie = f'const d = new Date(); d.setTime(d.getTime() + (365 * 24 * 60 * 60 * 1000)); const expires = "expires=" + d.toUTCString(); document.cookie = "cowboys={st.session_state.user + '&' + st.session_state.hash};" + expires + ";path=/";'
                        setcookie = f'const d = new Date(); d.setTime(d.getTime() + (365 * 24 * 60 * 60 * 1000)); const expires = "expires=" + d.toUTCString(); document.cookie = "cowboys={st.session_state.user + '&' + st.session_state.hash};" + expires'
                        streamlit_js_eval(js_expressions=setcookie)
                        reload = True
                    else:
                        st.write('Wrong password')
            if reload:
                showpage('sellers')
    
page = st.query_params.get_all('page')
if len(page)==0:
    page.append('sellers')
user = st.query_params.get_all('user')
if len(user)==0:
    user.append(' ')
hash = st.query_params.get_all('hash')
if len(hash)==0:
    hash.append(' ')
a = st.query_params.get_all('a')
if len(a)==0:
    a.append(' ')

if hash[0] == hashlib.sha256((user[0]+st.secrets['MYKEY']).encode()).hexdigest():
    st.session_state.user = user[0]
    st.session_state.hash = hash[0]
    st.session_state.auth = 'Y'
    #st_cookie.update('user')
    #st_cookie.update('hash')
    #setcookie = f'const d = new Date(); d.setTime(d.getTime() + (365 * 24 * 60 * 60 * 1000)); const expires = "expires=" + d.toUTCString(); document.cookie = "cowboys={st.session_state.user + '&' + st.session_state.hash};" + expires + ";path=/";'
    setcookie = f'const d = new Date(); d.setTime(d.getTime() + (365 * 24 * 60 * 60 * 1000)); const expires = "expires=" + d.toUTCString(); document.cookie = "cowboys={st.session_state.user + '&' + st.session_state.hash};" + expires'
    streamlit_js_eval(js_expressions=setcookie)
else:
    #st_cookie.apply()
#    if 'user' not in st.session_state:
    if False:
        cookie = streamlit_js_eval(js_expressions="document.cookie")
        if cookie is not None:
#        if True:
            if cookie.find('st-cookie-user=') > -1:
                user = cookie[cookie.find('st-cookie-user=')+len('st-cookie-user='):]
                user = user[:user.find(';')]
                st.session_state.user = user
                st_cookie.update('user')
            if cookie.find('st-cookie-hash=') > -1:
                hash = cookie[cookie.find('st-cookie-hash=')+len('st-cookie-hash='):]
                hash = hash[:hash.find(';')]
                st.session_state.hash = hash
                st_cookie.update('hash')
    cookie = streamlit_js_eval(js_expressions="document.cookie")
    if cookie is not None:
        if cookie.find('cowboys=') > -1:
            cowboys = cookie[cookie.find('cowboys=')+len('cowboys='):]
            st.session_state.user = cowboys[:cowboys.find('&')]
            hash = cowboys[cowboys.find('&')+1:]
            st.session_state.hash = hash[:hash.find(';')]
            #st.markdown(f'{st.session_state.user + " " + st.session_state.hash}', unsafe_allow_html=True)
#    if 'user' in st.session_state and 'hash' in st.session_state:
            if st.session_state.hash == hashlib.sha256((st.session_state.user+st.secrets['MYKEY']).encode()).hexdigest():
                st.session_state.auth = 'Y'
            else:
                st.session_state.auth = 'N'
                st.session_state.user = ''
                st.session_state.hash = ''
    else:
        st.session_state.auth = 'N'
        st.session_state.user = ''
        st.session_state.hash = ''

#st.markdown('''<!-- Google tag (gtag.js) --><script async src="https://www.googletagmanager.com/gtag/js?id=G-3T8LW0P2B2"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-3T8LW0P2B2');</script>''',unsafe_allow_html=True)
#st.markdown('<img src="./app/static/giants.jpg">', unsafe_allow_html=True)
st.title('Dallas Cowboys VETTED Season Ticket Holder Marketplace Web App')
showpage(page[0])






