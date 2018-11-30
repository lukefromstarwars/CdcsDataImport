from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from utils_analysis import *


# DbDefault.
# execute_sql()
CONN_STR = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=(localdb)\mssqllocaldb;DATABASE=CdcsData;Trusted_Connection=yes;"


def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		message = "Topic #%d: " % topic_idx
		message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
		print_info(message)
	print()


def get_organisation_keywords(df: DataFrame, stop_words: list):
	n_top_words = 100
	n_components = 10
	weight_threshold = 0.3

	# vectorizer = TfidfVectorizer(stop_words=stop_words)
	# vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_top_words, stop_words=stop_words, strip_accents='ascii')
	vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_top_words, stop_words=stop_words, strip_accents='ascii')
	vect = vectorizer.fit_transform(df['Text'])
	featured_names = vectorizer.get_feature_names()
	print_info(featured_names)
	print_info("vect.shape", vect.shape)
	print_info("vect", vect)
	vect_array = vect.toarray()
	print_info("vect_array", vect_array)
	print_info("vect_array", vect_array.shape)

	# nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(vect)
	# print_top_words(nmf, featured_names, n_top_words)

	# for f in vect.toarray():
	# 	print_info(f)
	# 	print_info(vect[f])

	for idx in vect_array:
		print(idx)
		feature_mask = (idx > weight_threshold)

		print(feature_mask)
	# print([f for f in vect_array[idx]])

	pass


def get_keywords():
	df = sql2df_from_conn("SELECT * from v_OrganisationTexts", CONN_STR)
	fr_cols = get_cols_with_suffix(df, '_FR')
	nl_cols = get_cols_with_suffix(df, '_NL')

	df = df.fillna('')

	df['Text'] = df[fr_cols].apply(" ".join, axis=1)
	# df['Text_NL'] = df[fr_cols].apply(" ".join, axis=1)

	stop_words = get_list_from_txt_file("stopWords_FR.txt") + get_list_from_txt_file("stopWords_NL.txt") + get_list_from_txt_file("stopWords_CUSTOM.txt")

	print_debug(df.shape)

	get_organisation_keywords(df[['Id', 'Text']], stop_words)


def get_xls_all_edges():
	df = read_pickle('df_edges').reset_index(drop=True)
	print_info(df.shape)

	df = df.dropna(subset=['SourceId'])
	print_info(df.shape)

	missing_targets = df['TargetId'].isnull()
	df.loc[missing_targets, 'Target_type'] = 'Organisation'
	df.loc[missing_targets, 'Target'] = df['Organisation']
	df.loc[missing_targets, 'TargetId'] = df['OrganisationId']
	df['TargetId'] = df['TargetId'].astype(int)
	df['SectorId'] = df['SectorId'].astype(int)
	df['SourceId'] = df['SourceId'].astype(int)
	# df['Label'] = df['Source_type']

	le = LabelEncoder()
	le.fit(df['Sector'])
	df['Label'] = le.transform(df['Sector'])
	df['modularity_class'] = le.transform(df['Sector'])
	df.reset_index(inplace=True)
	rename_column_to(df, 'index', 'id')

	print_info(df.dtypes)

	save_as_xlsx(df, 'all_edges')
	save_as_pickle(df, 'df_all_edges')


def get_xls_cat_org_edges():
	df = read_pickle('df_all_edges')
	df = df[df['Source_type'] == "Category"]
	df = df[df['Target_type'] == "Organisations"]

	save_as_xlsx(df, 'catOrg_edges')
	save_as_pickle(df, 'df_catOrg_edges')


def get_edges():
	df_h = read_pickle('df_organisations_hierarchies')
	rename_column_to(df_h, 'Name_FR', 'Organisation')
	rename_column_to(df_h, 'Id', 'OrganisationId')
	print_info(get_cols_alphabetically(df_h))

	cols = ['Sector', 'Category', 'Organisation', 'SectorId', 'CategoryId', 'OrganisationId']
	st_cols = ['Source', 'Target', 'SourceId', 'TargetId', 'Source_type', 'Target_type']
	df = DataFrame(columns=cols + st_cols)

	df.dropna(subset=['SectorId'], inplace=True)

	# Topic to topic
	for i in range(1, 4):
		new_cols = [f'T{i}', f'T{i+1}', f'Topic{i}Id', f'Topic{i+1}Id']
		print_info(new_cols)

		df_tmp = df_h[cols + new_cols]
		df_tmp['Source_type'] = 'Topic'
		df_tmp['Target_type'] = 'Topic'
		df_tmp.columns = cols + st_cols

		df = pd.concat([df, df_tmp], 0)

	# Sector to topic
	new_cols = ['Sector', 'T1', 'SectorId', 'Topic1Id']
	print_info(new_cols)

	df_tmp = df_h[cols + new_cols]
	df_tmp['Source_type'] = 'Sector'
	df_tmp['Target_type'] = 'Topic'
	df_tmp.columns = cols + st_cols

	df = pd.concat([df, df_tmp], 0)

	# Topic to category
	new_cols = ['T4', 'Category', 'Topic4Id', 'CategoryId']
	print_info(new_cols)

	df_tmp = df_h[cols + new_cols]
	df_tmp['Source_type'] = 'Topic'
	df_tmp['Target_type'] = 'Category'
	df_tmp.columns = cols + st_cols

	df = pd.concat([df, df_tmp], 0)

	# Category to organisation
	new_cols = ['Category', 'Organisation', 'CategoryId', 'OrganisationId']
	print_info(new_cols)

	df_tmp = df_h[cols + new_cols]
	df_tmp['Source_type'] = 'Category'
	df_tmp['Target_type'] = 'Organisation'
	df_tmp.columns = cols + st_cols

	df = pd.concat([df, df_tmp], 0)

	df['SectorId'].fillna(0, inplace=True)
	df['Sector'].fillna('Missing', inplace=True)
	df.loc[df['TargetId'] == '', 'Target'] = df['Organisation']
	df.loc[df['TargetId'] == '', 'TargetId'] = df['OrganisationId']

	# df['Source_type'].fillna('Missing', inplace=True)
	# df['Target_type'].fillna('Organisation', inplace=True)
	# df[df['SectorId'.isnull()]] = 'Missing'
	# df[df['Source_type'.isnull()]] = df['OrganisationId']
	# df[df['Target_type'.isnull()]] = 'Organisation'
	# df[df['TargetId'.isnull()]] = df['OrganisationId']
	# df[df['TargetId'.isnull()]] = df['OrganisationId']
	# df[df['TargetId'.isnull()]] = df['OrganisationId']

	save_as_pickle(df, 'df_edges')


def get_organisations_with_categories():
	df_organisations = sql2df_from_conn("SELECT Id, Name_FR, CategoriesIds  FROM Organisations", CONN_STR)

	print_info("df_organisations", get_cols_alphabetically(df_organisations))

	df_organisations["Categories"] = df_organisations["CategoriesIds"].str.split(",")

	cols = df_organisations.columns.tolist() + ['CategoryId']
	df = DataFrame(columns=cols)

	for index, row in df_organisations.iterrows():
		category_ids = [int(i) for i in row["Categories"] if len(i) > 0]

		# --
		print_debug(category_ids)

		# --
		if len(category_ids) > 0:
			for i in category_ids:
				row["CategoryId"] = i
				new_row = DataFrame([row.tolist()], columns=cols)
				df = pd.concat([df, new_row], 0)
	# df.append(new_row, ignore_index=True)

	df = remove_col_with_prefix(df, "Categories")
	df.reset_index(drop=True, inplace=True)

	save_as_pickle(df, 'df_organisations')

	print_debug("starting point")


def get_organisations_hierarchies():
	# DRIVER={ODBC Driver 17 for SQL Server};SERVER=test;DATABASE=test;UID=user;PWD=password

	df_hierarchies = sql2df_from_conn("SELECT * FROM v_Detailed_Topics", CONN_STR)
	df_organisations = read_pickle('df_organisations')
	df_organisations['CategoryId'] = pd.to_numeric(df_organisations['CategoryId'])
	print_info(get_cols_alphabetically(df_organisations))

	df_organisations = df_organisations.dropna(subset=['CategoryId'])
	print_info("df_hierarchies", df_hierarchies.dtypes)
	print_info("df_organisations", df_organisations.dtypes)

	df = df_organisations.merge(df_hierarchies, on='CategoryId', how='left')

	save_as_pickle(df, 'df_organisations_hierarchies')
