import pandas_gbq
import pandas as pd

# name of google BIG QUERY project
project_id = 'gymnos-243001'

def getTime():
    return pd.date_range('now', periods=1)


def insert(value):
    # perform insert into database
    pandas_gbq.to_gbq(value, 'gymnos.records', project_id=project_id, if_exists='append')
    print('done')


def getAll():
    # loads database
    df = pandas_gbq.read_gbq('SELECT * FROM gymnos.records', project_id=project_id, dialect="legacy")
    # retrieve values
    return df.head()


def main():
    # gets current time
    # first get when the person enters frame using pd.date_range('now', periods=1)
    # then when they exit frame get another value using pd.date_range('now', periods=1)
    value = pd.DataFrame({
        'in_frame': getTime(),
        'out_frame': getTime()
    })

    insert(value)


if __name__ == '__main__':
    main()
