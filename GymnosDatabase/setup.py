import pandas_gbq
import pandas as pd
#%matplotlib inline

#name of google BIG QUERY project
project_id = 'gymnos-243001'

#loads database
#df = pandas_gbq.read_gbq('SELECT * FROM gymnos.records', project_id=project_id, dialect="legacy")

#gets current time
#first get when the person enters frame using pd.date_range('now', periods=1)
#then when they exit frame get another value using pd.date_range('now', periods=1)
value = pd.DataFrame({
    'in_frame': pd.date_range('now', periods=1),
    'out_frame': pd.date_range('now', periods=1)
})

#perform insert into database
pandas_gbq.to_gbq(value, 'gymnos.records', project_id=project_id, if_exists='append')

#retrieve values
#print(df.head())

#df.Mobile_app_installs.plot.scatter()
