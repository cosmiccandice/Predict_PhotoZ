import os

import pandas as pd
import pandas.io.sql as sqlio
import psycopg2



##################################################
#
# Info to connect to DB
#
_db_user = os.getenv('transients_db_user', None)
_db_info = os.getenv('transients_db_info', None)

_db_host = 'transients.ciera.northwestern.edu'
_db_port = 5433

_default_schema = 'public'


##################################################
#
# Define common database class
#     These will be inherited for future database instances.
#     This is to reduce duplicate efforts when defining databases.
#

class CommonDatabase():

    #
    # Initialize
    #
    def __init__(self, db_name=None, db_host = _db_host, db_user = _db_user, db_info = _db_info, db_port = _db_port):

        self.db_name = db_name
        self.db_host = db_host
        self.db_user = db_user
        self.db_info = db_info
        self.db_port = db_port

        if self.db_name is not None:
            self.conn = psycopg2.connect(f"host={self.db_host} dbname={self.db_name} user={self.db_user} password={self.db_info} port={self.db_port}")
            self.cur = self.conn.cursor()
    
    
    def close(self):
        self.cur.close()
        self.conn.close()


    def commit(self):

        self.conn.commit()


    def execute(self, query, query_params = None):
        
        if query_params is not None:
            errors = self.cur.execute(query, query_params)
        else:
            errors = self.cur.execute(query)
        if errors is not None:
            print("Problem running this query.")
            print(errors)


    def fetchall(self):
        return self.cur.fetchall()


    def fetchone(self):
        return self.cur.fetchone()


    # Check that the connection to the database is working
    def check_connection(self):

        try:
            self.execute('SELECT TRUE;')
            return True
        except:
            return False


    def describe_table(self, table_name, schema='public'):

        query = 'SELECT * FROM information_schema.columns WHERE table_schema = %s AND table_name = %s ORDER BY column_name'
        return sqlio.read_sql_query(query, self.conn, params=[schema, table_name,])

    
    def list_tables(self, schema='public'):
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_type='BASE TABLE'"
        return sqlio.read_sql_query(query, self.conn, params=[schema,])


    def get_table_info_from_nearby_sources_by_coordinates(self, table_name, parameter_list, ra_degrees, decl_degrees, angular_separation_degrees = 2./3600., schema=_default_schema, limit = None, order_by='angular_separation ASC'):
        
        # Take the column list and first append an angular distance field
        column_list = parameter_list.copy()
        column_list.append("q3c_dist(ra, decl, %s, %s) * 3600. as angular_separation")

        # Create query to run
        query = (f"SELECT {','.join(column_list)} "
                 f"FROM {schema}.{table_name} WHERE "
                 "q3c_radial_query(ra, decl, %s, %s, %s) "
                 f'ORDER BY {order_by}')
        if limit is not None and efdim.is_numeric(limit):
            query += f' LIMIT {limit}'


        # Format values into a simple tuple
        param_list = (ra_degrees, decl_degrees, ra_degrees, decl_degrees, angular_separation_degrees)

        return sqlio.read_sql_query(query, self.conn, params=param_list)


    def get_table_info_from_nearby_extended_source_from_coordinates(self, table_name, parameter_list, ra_degrees, decl_degrees, maj_axis, min_axis, pa = 0., schema=_default_schema, limit = None, order_by='angular_separation ASC'):
        
        # Take the column list and first append an angular distance field
        column_list = parameter_list.copy()
        column_list.append("q3c_dist(ra, decl, %s, %s) * 3600. as angular_separation")

        # Create query to run
        query = f"SELECT {','.join(column_list)} " + \
                f"FROM {schema}.{table_name} WHERE " + \
                 "q3c_ellipse_query(ra, decl, %s, %s, " + f"{maj_axis}/3600., {min_axis}/{maj_axis}, {pa}) " + \
                f'ORDER BY {order_by}'
        if limit is not None and efdim.is_numeric(limit):
            query += f' LIMIT {limit}'


        # Format values into a simple tuple
        param_list = (ra_degrees, decl_degrees, ra_degrees, decl_degrees)

        return sqlio.read_sql_query(query, self.conn, params=param_list)


##################################################
#
# Info to connect to WISE DB
#
class WiseDatabase(CommonDatabase):
    

    def __init__(self):
        super().__init__(db_name='wise')    


def match_sdss_to_unwise():
        
    sdss = pd.read_csv('sdss17_to_ps1dr2_xmatch.csv', sep=',', low_memory=False)
    
    Wise = WiseDatabase()
    db_info = Wise.describe_table('unwise')
    

    existing_cols = sdss.columns.to_list()
    unwise_cols = [f'unwise_{x}' for x in db_info.column_name.to_list() + ['angular_separation']]
    header = ','.join(existing_cols + unwise_cols)
    
    with open('sdss17_and_ps1dr2_to_unwise_xmatch.csv', 'w') as f_out:
    
        f_out.write(header + '\n')
    
        for i, row in sdss.iterrows():
      
            xmatches = Wise.get_table_info_from_nearby_sources_by_coordinates('unwise', 
                       db_info.column_name.to_list(),
                       row['ps_ra'], 
                       row['ps_decl'],
                       1.9/3600.)
        
            if xmatches.size > 0:
                
                sdss_out = ','.join([str(x) for x in row.to_list()])
                unwise_out = ','.join([str(x) for x in xmatches.iloc[0].to_list()])
                
                f_out.write(f'{sdss_out},{unwise_out}\n')
                

def main():

    match_sdss_to_unwise()


if __name__ == "__main__":
    main()
