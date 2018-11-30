import functools
import os
import pyodbc
import re
import sys
import time
import warnings
from datetime import timedelta
from importlib import reload
from itertools import combinations
from numbers import Number

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors, cm
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from unidecode import unidecode


# TO REMOVE 18.05.2018
# from datetime import timedelta
# from openpyxl import load_workbook
# END REMOVE

# cols = get_cols_alphabetically(df)
# for col in cols:
#     print('"{}" : "{}",'.format(col, col.lower()))

# -- Settings
# ------------------------------------------------------------------------------------------
# dataframe


# graphics
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 15)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 160)
pd.set_option('display.precision', 2)
pd.set_option('display.large_repr', 'truncate')
pd.set_option('display.expand_frame_repr', True)

sns.set(style='whitegrid', color_codes=True)
hf_pal = sns.color_palette(['#3F9CB2', '#FF89B3'])
sns.set_palette(hf_pal)
sns.set_style({'font.sans-serif': 'Calibri',
			   'axes.grid'      : True})

sns.set_context(font_scale=.9)


# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Calibri', 'Tahoma']

class MyVars:
	TIME_STAMP = "%Y%m%d_%H%M%S"


class MyFolders:
	""""
	Project folders
	"""

	ML = 'ML/'
	GRAPHS = 'GRAPHS/'
	CLUSTER_GRAPHS = 'GRAPH_CLUSTERS/'
	EXCEL = 'EXCEL/'
	SOURCES = 'SOURCES/'
	PICKLES = 'PICKLES/'
	HTML = 'HTML/'
	UTILS = 'UTILS/'


class MyPrint:
	CHECK = 'CHECK > '
	DURATION = 'DURATION > '
	INFO = 'INFO > '
	DEBUG = 'DEBUG > '
	INPUT = 'INPUT_PROMPT > '
	TEST = 'TEST > '
	WARNING = 'WARNING > '


class MyClusters:
	# --
	MISS_CLASSIFIED_PCT = 'MISS_CLASSIFIED_PCT'
	WORST_ASSIGNMENT = 'WORST_ASSIGNMENT'
	CLUSTER = 'CLUSTER'
	CLUSTER_NAME = 'CLUSTER_NAME'
	CLUSTER_TOPIC = 'CLUSTER_TOPIC'
	CLUSTERING_ALGORITHM = 'CLUSTERING_ALGORITHM'
	MAX_NCOMPS = 'MAX_NCOMPS'
	METRIC = 'METRIC'
	MIN_NCOMPS = 'MIN_NCOMPS'
	N_CLUSTERS = 'N_CLUSTERS'
	N_COMPS = 'N_COMPS'
	SILHOUETTE_AVG = 'SILHOUETTE_AVG'
	NB_HISTORICAL_YEARS = 4
	NB_MODELING_YEARS = 3

	# -- ALGO
	ALGO_KMeans = 'KMeans'
	ALGO_MiniBatchKMeans = 'MiniBatchKMeans'
	ALGO_MeanShift = 'MeanShift'
	ALGO_Spectral = 'Spectral'
	ALGO_Ward = 'Ward'
	ALGO_Agglomerative = 'Agglomerative'
	ALGO_Birch = 'Birch'

	# -- METRIC
	METRIC_CityBlock = 'cityblock'
	METRIC_Cosine = 'cosine'
	METRIC_Euclidean = 'euclidean'
	METRIC_L1 = 'l1'
	METRIC_L2 = 'l2'
	METRIC_Manhattan = 'manhattan'


class MyCols:
	"""
	Standard columns names in db
	"""
	# project columns
	CAT = 'CAT'
	CODE = 'VAR_CODE'
	COL_NAME = 'COL_NAME'
	CURRENT_DATA = 'CURRENT_DATA'
	DESC = 'VAR_DESC'
	EN = 'EN'
	FACTOR = 'FACTOR'
	FACTOR_FR = 'FACTOR_FR'
	FR = 'FR'
	LST = 'VAR_LST'
	NAME = 'VAR_NAME'
	NAMES = 'LST_NAME'
	NL = 'NL'
	ORIGINAL = 'ORIGINAL'
	PKL_NAME = 'PKL_NAME'
	SUMMARY = 'SUMMARY'
	TYPE = 'VAR_TYPE'
	USED = 'VAR_USED'
	VALUE = 'VALUE'
	VAR = 'VAR'

	# column types
	BIN = 'BIN'
	COUNT = 'COUNT'
	NUM = 'NUM'
	ORD = 'ORD'
	STATUS = 'STATUS'
	NULL_VALUES = 'NULL_VALUES'

	# general columns
	CORR = 'CORR'
	ENCODING_YEAR = "NR_ENC_YR"
	INDEX = 'INDEX'
	LENGTH = 'LENGTH'
	STRENGTH = 'STRENGTH'
	MEAN = 'MEAN'
	MEDIAN = 'MEDIAN'
	NOT_AVAILABLE = 'N/A'
	PERCENT = 'PERCENTAGE'
	CUM_SUM = 'CUMUL'
	PERCENT_FILLED = 'PERCENT_W_VALUE'
	PERCENT_NA = 'PERCENT_NA'
	RANKING = 'RK'
	STATS = 'STATS'
	STDV = 'STDV'
	TMP = 'TMP'
	TOTAL = 'TOTAL'
	VARIANCE_COEFF = 'CV'

	# vars
	YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
	NISS = 'NISS'

	# colorMaps
	EXCEL_COLORMAP = colors.ListedColormap(
		['#4F81BD', '#C0504D', '#9BBB59', '#8064A2', '#4BACC6', '#F79646', '#2C4D75', '#772C2A', '#5F7530', '#276A7C'])
	EXCEL_COLORMAP_2 = colors.ListedColormap(
		['#4F81BD', '#9BBB59', '#4BACC6', '#2C4D75', '#5F7530', '#276A7C', '#729ACA', '#AFC97A', '#6FBDD1', '#3A679C'])
	COLORMAP = cm.get_cmap('viridis')


class MyDfs:
	DATA_ALL = 'dataAll'
	DATA_NO_NISS = 'dataNoNiss'
	DATA_NEW = 'dataNew'
	DATA_CENTER_NEW = 'dataCenterNew'
	DATA_PROGRAM_NEW = 'dataProgramNew'
	DATA_FIRST_VISIT = 'dataFirstVisit'
	DATA_YR_FIRST_VISIT = 'dataYrFirstVisit'
	KNN_COLS = [MyClusters.CLUSTER_TOPIC,
				MyClusters.CLUSTER_NAME,
				MyClusters.N_CLUSTERS,
				MyClusters.N_COMPS,
				MyClusters.SILHOUETTE_AVG,
				MyClusters.METRIC,
				MyClusters.CLUSTERING_ALGORITHM,
				MyClusters.WORST_ASSIGNMENT,
				MyClusters.MISS_CLASSIFIED_PCT]

	FrenchDict = {
		'AGE_GROUP'                                                      : "Groupe d'âge",
		'BIN_FIRST_VISIT'                                                : "Prem. visite",
		'BIN_HAS_EDUCATION'                                              : "Education",
		'BIN_HAS_FREQ_USE'                                               : "Usage",
		'BIN_HAS_MAIN_PROD'                                              : "Produit princ.",
		'BIN_HAS_NISS'                                                   : "NISS",
		'BIN_HEALTH_ISSUES'                                              : 'Arrêt maladie/Invalidité',
		'BIN_ILLEGAL_PROD'                                               : "Produit illégal",
		'BIN_INCOME_STAT_FR_ALLOCATION_DE_CHOMAGE'                       : "Allocations de chômage",
		'BIN_INCOME_STAT_FR_ALLOCATIONS_FAMILIALES'                      : "Allocations Familiales",
		'BIN_INCOME_STAT_FR_AUCUN_REVENU_PROPRE'                         : "Sans source de revenus",
		'BIN_INCOME_STAT_FR_BOURSE_D_ETUDES'                             : "Bourse d'étude",
		'BIN_INCOME_STAT_FR_INCONNU'                                     : "Source de revenus inconnue",
		'BIN_INCOME_STAT_FR_INDEMNITE_POUR_MALADIE_OU_INVALIDITE'        : "Indémnité pour maladie ou invalidité",
		'BIN_INCOME_STAT_FR_REVENU_MINIMUM_OU_SUPPORT_DU_CPAS'           : "Revenu min./CPAS",
		'BIN_INCOME_STAT_FR_SALAIRE_REVENUS_DU_TRAVAIL'                  : "Revenus du travail",
		'BIN_IS_AMBULATORY'                                              : "Ambulatoire",
		'BIN_IS_BE_NEW'                                                  : "Nouveau patient (BE)",
		'BIN_IS_CENT_NEW'                                                : "Nouv. df institution",
		'BIN_IS_MALE'                                                    : "H/F",
		'BIN_IS_NEW'                                                     : "Nouveau",
		'BIN_IS_PROG_NEW'                                                : "Nouv. ds service",
		'BIN_LABOUR_STAT_FR_AU_CHOMAGE'                                  : "Chômage",
		'BIN_LABOUR_STAT_FR_ECOLIER_ETUDIANT_EN_FORMATION'               : "Etudiant",
		'BIN_LABOUR_STAT_FR_EMPLOI_OCCASIONNEL'                          : "Emploi occasionnel",
		'BIN_LABOUR_STAT_FR_EMPLOI_REGULIER'                             : "Emploi régulier",
		'BIN_LABOUR_STAT_FR_HOMME_FEMME_AU_FOYER'                        : "H/F au foyer",
		'BIN_LABOUR_STAT_FR_INCAPACITE_DE_TRAVAIL'                       : "En incapacité de travail",
		'BIN_LABOUR_STAT_FR_INCONNU'                                     : "Situation de travail inconnue",
		'BIN_LAST_VISIT'                                                 : "Dern. visite",
		'BIN_LVN_SDF'                                                    : "Sans domicile fixe",
		'BIN_LVN_STAT_WHERE_FR_DANS_DES_LOGEMENTS_VARIABLES'             : "Habite log. variable",
		'BIN_LVN_STAT_WHERE_FR_DANS_LA_RUE'                              : "En rue",
		'BIN_LVN_STAT_WHERE_FR_DANS_UN_AUTRE_TYPE_D_ENDROIT'             : "Vit ailleurs",
		'BIN_LVN_STAT_WHERE_FR_DANS_UN_DOMICILE_FIXE'                    : "A un domicile fixe",
		'BIN_LVN_STAT_WHERE_FR_EN_INSTITUTION'                           : "En institution",
		'BIN_LVN_STAT_WHERE_FR_EN_PRISON'                                : "En prison",
		'BIN_LVN_STAT_WHERE_FR_INCONNU'                                  : "Domicile inconnu",
		'BIN_LVN_STAT_WITH_WHOM_FR_AUTRE'                                : "Hab. avec qqn",
		'BIN_LVN_STAT_WITH_WHOM_FR_AVEC_DES_AMIS_OU_AUTRES_PERSONNES'    : "Hab. avec des amis",
		'BIN_LVN_STAT_WITH_WHOM_FR_AVEC_DES_AUTRES_MEMBRES_DE_MA_FAMILLE': "Hab. avec famille",
		'BIN_LVN_STAT_WITH_WHOM_FR_AVEC_UN_MES_PARENT'                   : "Hab. avec parents",
		'BIN_LVN_STAT_WITH_WHOM_FR_EN_COUPLE'                            : "Vit en couple",
		'BIN_LVN_STAT_WITH_WHOM_FR_INCONNU'                              : "Hab. (inconnu)",
		'BIN_LVN_STAT_WITH_WHOM_FR_N_A'                                  : "Cohabitation inconnue",
		'BIN_LVN_STAT_WITH_WHOM_FR_SEUL'                                 : "Seul",
		'BIN_MAIN_PROD_ALCOHOL'                                          : "Alcool",
		'BIN_MAIN_PROD_CANNABIS'                                         : "Cannabis",
		'BIN_MAIN_PROD_COCAINE'                                          : "Cocaïne",
		'BIN_MAIN_PROD_HALLUCINOGENS'                                    : "Hallucinogènes",
		'BIN_MAIN_PROD_HYPNOTICS'                                        : "Hypnotiques",
		'BIN_MAIN_PROD_OPIATES'                                          : "Opiacés",
		'BIN_MAIN_PROD_OTHER'                                            : "Autres produits",
		'BIN_MAIN_PROD_STIMULANTS'                                       : "Autres stimulants",
		'BIN_MAIN_PROD_VOLATILES'                                        : "Inhalants",
		'BIN_NATIONALITY_FR_BELGE'                                       : "Belge",
		'BIN_NATIONALITY_FR_INCONNU'                                     : "Nationalité inconnue",
		'BIN_NATIONALITY_FR_N_A'                                         : "Nationalité (N/A)",
		'BIN_NATIONALITY_FR_NON_BELGE_NON_UNION_EUROPEENNE'              : "Hors UE",
		'BIN_NATIONALITY_FR_NON_BELGE_UNION_EUROPEENNE'                  : "UE",
		'BIN_PREV_TREAT_INCONNU'                                         : "Traitement préc. (inconnu)",
		'BIN_PREV_TREAT_NON'                                             : "Pas de traitement préc. ",
		'BIN_PREV_TREAT_OUI'                                             : "Traitement préc.",
		'BIN_PROD_ALCOHOL'                                               : "Alcool",
		'BIN_PROD_AMPHETAMINES'                                          : "Amphétamines",
		'BIN_PROD_BARBITURATES'                                          : "Barbituriques",
		'BIN_PROD_BENZODIAZEPINES'                                       : "Benzodiazépines",
		'BIN_PROD_BUPRENORPHINE'                                         : "Buprenorphine",
		'BIN_PROD_COCAINE'                                               : "Cocaïne",
		'BIN_PROD_CRACK'                                                 : "Crack",
		'BIN_PROD_FENTANYL'                                              : "Fentanyl",
		'BIN_PROD_GHB'                                                   : "GHB",
		'BIN_PROD_HALLUCINOGENS'                                         : "Hallucinogènes",
		'BIN_PROD_HASH'                                                  : "Hash",
		'BIN_PROD_HEROIN'                                                : "Héroïne",
		'BIN_PROD_HYPNOTICS'                                             : "Hypnotiques",
		'BIN_PROD_KETAMINE'                                              : "Kétamine",
		'BIN_PROD_LSD'                                                   : "LSD",
		'BIN_PROD_MARIJUANA'                                             : "Marijuana",
		'BIN_PROD_MDMA'                                                  : "MDMA",
		'BIN_PROD_MEPHEDRONE'                                            : "Méphédrone",
		'BIN_PROD_METHADONE'                                             : "Méthadone",
		'BIN_PROD_METHAMPHETAMINES'                                      : "Méthamphetamines",
		'BIN_PROD_OPIATES'                                               : "Opiacés",
		'BIN_PROD_OTHER_CANNABIS'                                        : "Cannabis (Autre)",
		'BIN_PROD_OTHER_COCAINE'                                         : "Cocaïne (Autre)",
		'BIN_PROD_OTHER_HALLUCINOGENS'                                   : "Hallucinogènes (Autre)",
		'BIN_PROD_OTHER_HYPNOTICS'                                       : "Hypnotiques (Autre)",
		'BIN_PROD_OTHER_OPIATES'                                         : "Opiacés (Autre)",
		'BIN_PROD_OTHER_PROD'                                            : "Autres produits (Autre)",
		'BIN_PROD_OTHER_STIMULANTS'                                      : "Autres stimulants",
		'BIN_PROD_POWDER_COCAINE'                                        : "Cocaïne (poudre)",
		'BIN_PROD_STIMULANTS'                                            : "Stimulants",
		'BIN_PROD_VOLATILE'                                              : "Inhalants",
		'BIN_REFERRAL_FR_AUTRE'                                          : "Orienté - Autre",
		'BIN_REFERRAL_FR_INCONNU'                                        : "Réf. inconnu",
		'BIN_REFERRAL_FR_LA_JUSTICE'                                     : "Réf. justice",
		'BIN_REFERRAL_FR_MOI_MEME'                                       : "Décision personnelle",
		'BIN_REFERRAL_FR_QUELQU_UN_DE_MA_FAMILLE'                        : "Réf. famille",
		'BIN_REFERRAL_FR_UN_AMI'                                         : "Réf. ami(e)",
		'BIN_REFERRAL_FR_UN_AUTRE_SERVICE_MEDICAL_OU_PSYCHOSOCIAL'       : "Réf. serv psy/med/soc.",
		'BIN_REFERRAL_FR_UN_CENTRE_POUR_TOXICOMANES'                     : "Réf. centre pr tox.",
		'BIN_REFERRAL_FR_UN_HOPITAL'                                     : "Réf. hôpital",
		'BIN_REFERRAL_FR_UN_MEDECIN_GENERALISTE'                         : "Réf. médecin",
		'BIN_TYPE_OF_PROGRAM_FR_BAS_SEUIL'                               : "Bas-seuil",
		'BIN_TYPE_OF_PROGRAM_FR_CONSULTATIONS_SPECIALISEES'              : "Consultations spécialisées",
		'BIN_TYPE_OF_PROGRAM_FR_HOPITAL_GENERAL'                         : "Hôpital général",
		'BIN_TYPE_OF_PROGRAM_FR_HOPITAL_PSYCHIATRIQUE'                   : "Hôpital psychiatrique",
		'BIN_TYPE_OF_PROGRAM_FR_SERVICE_SPECIALISE_EN_MILIEU_CARCERAL'   : "Service spécialisé en milieu carcéral",
		'BIN_USE_PROD_CANNABIS'                                          : "Cannabis",
		'BIN_USE_PROD_COCAINE'                                           : "Cocaïne",
		'BIN_USE_PROD_HALLUCINOGENS'                                     : "Hallucinogènes",
		'BIN_USE_PROD_HYPNOTICS'                                         : "Hypnotiques & sédatifs",
		'BIN_USE_PROD_OPIATES'                                           : "Opiacés",
		'BIN_USE_PROD_STIMULANTS'                                        : "Stimulants",
		'BIN_USE_PROD_VOLATILES'                                         : "Inhalants",
		'BIN_WORK_FR_HEALTH_ISSUES'                                      : "Arrêt mal./Invalidité",
		'BIN_WORK_HEALTH_ISSUES'                                         : "Problèmes de santé",
		'BIN_WORK_IS_WORKING'                                            : "Emploi",
		'BIN_WORK_OTHER'                                                 : "Emploi (Autre)",
		'BIN_WORK_REGULARLY'                                             : "Emploi régulier",
		'BIN_WORK_RETIRED'                                               : "Pensionné",
		'BIN_WORK_UNEMPLOYED'                                            : "Chômage",
		'BIN_WORK_UNKNOWN'                                               : "Situation de travail inconnue",
		'BIN_YR_FIRST_VISIT'                                             : "Prem. visite (an)",
		'BIN_YR_LAST_VISIT'                                              : "Dern. visite (an)",
		'CAT_AMBURESIDENTIAL_FR'                                         : "Ambulatoire/Résidentiel",
		'CAT_EDUCATION_LEVEL'                                            : "Niveau d'éducation",
		'CAT_EDUCATION_LEVEL_FR'                                         : "Niveau d'éducation",
		'CAT_FREQ_USE'                                                   : "Usage (fréq.)",
		'CAT_FREQ_USE_FR'                                                : "Usage (fréq.)",
		'CAT_INCOME_STAT'                                                : "Revenus",
		'CAT_INCOME_STAT_FR'                                             : "Revenus",
		'CAT_LABOUR_STAT'                                                : "Emploi",
		'CAT_LABOUR_STAT_FR'                                             : "Emploi",
		'CAT_LVN_CHILDREN'                                               : "Enfants à charge",
		'CAT_LVN_CHILDREN_FR'                                            : "Enfants à charge",
		'CAT_LVN_STAT_WHERE'                                             : "Vit (où)",
		'CAT_LVN_STAT_WHERE_FR'                                          : "Vit (où)",
		'CAT_LVN_STAT_WITH_WHOM'                                         : "Vit (avec)",
		'CAT_LVN_STAT_WITH_WHOM_FR'                                      : "Vit (avec)",
		'CAT_MAIN_PROD'                                                  : "Produit",
		'CAT_MAIN_PROD_FR'                                               : "Produit",
		'CAT_NATIONALITY'                                                : "Nationalité",
		'CAT_NATIONALITY_FR'                                             : "Nationalité",
		'CAT_PREV_TREAT'                                                 : "Traitement préc.",
		'CAT_PREV_TREAT_FR'                                              : "Traitement préc.",
		'CAT_PROD_CATEGORY'                                              : "Type de produit",
		'CAT_PROD_CATEGORY_FR'                                           : "Type de produit",
		'CAT_REFERRAL'                                                   : "Référent",
		'CAT_REFERRAL_FR'                                                : "Référent",
		'CAT_REG'                                                        : "Région",
		'CAT_REG_FR'                                                     : "Région",
		'CAT_SEX'                                                        : "Sexe",
		'CAT_SEX_FR'                                                     : "Sexe",
		'CAT_TYPE_OF_PROGRAM'                                            : "Type de service",
		'CAT_TYPE_OF_PROGRAM_FR'                                         : "Type de service",
		'CENTER'                                                         : "Centre",
		'CLUSTER'                                                        : "Groupe",
		'COUNT_PATIENT_CENTER_TREATMENTS'                                : "Nombre de traitements/Centre (Total)",
		'COUNT_PATIENT_PROGRAM_TREATMENTS'                               : "Nombre de traitements/Service (Total)",
		'COUNT_TREATMENTS'                                               : "Nombre de traitements (Total)",
		'COUNT_TREATMENTS_TO_DATE'                                       : "Nombre de traitements",
		'COUNT_YR_PATIENT_CENTER_TREATMENTS'                             : "Nombre de traitements/Centre (Total/An)",
		'COUNT_YR_PATIENT_PROGRAM_TREATMENTS'                            : "Nombre de traitements/Service (Total/An)",
		'COUNT_YR_TREATMENTS'                                            : "Nombre de traitements (Total/An)",
		'COUNT_YR_TREATMENTS_TO_DATE'                                    : "Nombre de traitements précédents",
		'DT_CENT_PREV_START_TREAT'                                       : "Préc.traitement (centre)",
		'DT_PREV_START_TREAT'                                            : "Préc.traitement",
		'DT_PROG_PREV_START_TREAT'                                       : "Préc.traitement (service)",
		'DT_START_TREAT'                                                 : "Date",
		'FIRST_USE_AGE_GROUP'                                            : "Groupe d'âge (prem. fois)",
		'FROM_WA'                                                        : "Venu(e) d'un centre wallon",
		'GAP_BETWEEN_CENTER_TREATMENTS_IN_DAYS'                          : "Nb. de jours sans trait. (centre)",
		'GAP_BETWEEN_PROGRAM_TREATMENTS_IN_DAYS'                         : "Nb. de jours sans trait. (service)",
		'GAP_BETWEEN_TREATMENTS_IN_DAYS'                                 : "Nb. de jours sans trait.",
		'IDC_PAT_CODED'                                                  : "ID",
		'IDN_ARROND_CENT'                                                : "Centre (ID)",
		'IDN_ARROND_CENT_FR'                                             : "Centre (CP)",
		'MY_IX'                                                          : "MY_IX",
		'NUM_AGE'                                                        : "Age",
		'NUM_AGE_FIRST_USE'                                              : "Age (prem. fois)",
		'NUM_FIRST_USE_TREATMENT_SPAN_IN_YEARS'                          : "Consomme depuis (en années)",
		'NUM_ILLEGAL_PROD_TYPES'                                         : "Type de prod. illégaux (N)",
		'NUM_ILLEGAL_PRODS'                                              : "Prod. illégaux (N)",
		'NUM_PROD_CANNABIS'                                              : "Cannabis (N)",
		'NUM_PROD_COCAINE'                                               : "Cocaîne (N)",
		'NUM_PROD_HALLUCINOGENS'                                         : "hallucinogènes (N)",
		'NUM_PROD_HYPNOTICS'                                             : "Hypnotiques (N)",
		'NUM_PROD_OPIATES'                                               : "Opiacés (N)",
		'NUM_PROD_STIMULANTS'                                            : "Stimulants (N)",
		'NUM_PROD_TYPES'                                                 : "Autres produits (N)",
		'NUM_PROD_VOLATILES'                                             : "Inhalants (N)",
		'ORD_AGE_GROUP'                                                  : "Groupe d'âge",
		'ORD_EDUCATION'                                                  : "Niveau d'éducation",
		'ORD_FIRST_USE_AGE_GROUP'                                        : "Groupe d'âge (prem.fois)",
		'ORD_FREQ_USE'                                                   : "Usage (fréq.)",
		'POST_PSY_BXL1'                                                  : "Parti(e) vers un autre centre bruxellois",
		'PRE_PSY_BXL1'                                                   : "Venu(e) d'un autre centre bruxellois",
		'PRE_RES_WA1'                                                    : "Venu(e) d'un service résidentiel wallon",
		'PROGRAM'                                                        : "Service",
		'YR_TREATMENT'                                                   : "Année"
	}


# region CODING TOOLS

def get_methods_n_attributes(obj):
	"""
	Print list of python obj methods and attributes
	:param obj:
	:return:
	"""
	print([method for method in dir(obj) if callable(getattr(obj, method))])


def import_or_reload(module_name, *names):
	import sys


	if module_name in sys.modules:
		reload(sys.modules[module_name])
	else:
		__import__(module_name, fromlist=names)

	for name in names:
		globals()[name] = getattr(sys.modules[module_name], name)


def deprecated(func):
	"""This is a decorator which can be used to mark functions
	as deprecated. It will result in a warning being emmitted
	when the function is used."""

	@functools.wraps(func)
	def new_func(*args, **kwargs):
		warnings.simplefilter('always', DeprecationWarning)  # turn off filter
		warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
		warnings.simplefilter('default', DeprecationWarning)  # reset filter
		return func(*args, **kwargs)

	return new_func


# endregion


# region PRINT_CONSOLE
# ------------------------------------------------------------------------------------------
def set_basic_pd_options():
	pd.set_option('display.max_columns', 15)
	pd.set_option('display.max_colwidth', 15)
	pd.set_option('display.max_rows', 50)
	pd.set_option('display.width', 160)
	pd.set_option('display.precision', 2)
	pd.set_option('display.large_repr', 'truncate')
	pd.set_option('display.expand_frame_repr', False)


def print_full(df, nb_rows=None, nb_columns=None):
	# Set full pandas options
	pd.set_option('display.max_rows', nb_rows)
	pd.set_option('display.max_columns', nb_columns)
	pd.set_option('display.max_colwidth', 30)
	pd.set_option('display.precision', 3)
	pd.set_option('display.expand_frame_repr', True)
	pd.set_option('display.large_repr', 'truncate')

	print(df)

	# Reset pandas options
	set_basic_pd_options()


def print_to_console(args, nb_rows=None, nb_columns=None, print_type=False):
	pd.set_option('display.max_rows', nb_rows)
	pd.set_option('display.max_columns', nb_columns)

	if print_type:
		for arg in args:
			if type(arg) == list:
				for el in arg:
					print(el, type(el))
			else:
				print(arg, type(arg))
		print('\n')
	else:
		for arg in args:
			if type(arg) == list:
				for el in arg:
					print(el)
			else:
				print(arg)
		print('\n')

	# Reset pandas options
	set_basic_pd_options()


def print_debug(*args, nb_rows=None, nb_columns=None, spaced=1, print_type=False):
	IN_DEBUG_MODE = (sys.gettrace() is not None)

	if IN_DEBUG_MODE:
		print('\n' * spaced, MyPrint.DEBUG)
		print_to_console(args, nb_rows, nb_columns, print_type)


def print_test(*args, nb_rows=None, nb_columns=None, spaced=1, print_type=False):
	print('\n' * spaced, MyPrint.TEST)
	print_to_console(args, nb_rows, nb_columns, print_type)


def print_warn(*args, nb_rows=None, nb_columns=None, spaced=1, print_type=False):
	print('\n' * spaced, MyPrint.WARNING)
	print_to_console(args, nb_rows, nb_columns, print_type)


def print_check(*args, nb_rows=None, nb_columns=None, spaced=1, print_type=False):
	print('\n' * spaced, MyPrint.CHECK)
	print_to_console(args, nb_rows, nb_columns, print_type)


def print_input(*args):
	print(MyPrint.INPUT)


def print_info(*args, nb_rows=None, nb_columns=None, spaced=1, print_type=False):
	print('\n' * spaced, MyPrint.INFO)
	print_to_console(args, nb_rows, nb_columns, print_type)


# endregion


# region FILE SAVE
# ------------------------------------------------------------------------------------------
def get_list_from_txt_file(fn: str):
	text_file = open(fn, "r")
	lines = text_file.read().split('\n')
	return lines


def get_file_path_and_name(str):
	file_path, file_name = os.path.split(str)
	return file_path, file_name


def create_dir_if_exists_not(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)


def reset_enum_file(fn):
	pkl = 'DbState/{}_state'.format(fn)
	init_state = 'INIT STATE - NO PICKLE YET'
	df = DataFrame([[MyCols.CURRENT_DATA, init_state]], columns=[MyCols.COL_NAME, MyCols.PKL_NAME])
	write_enums_to_file(fn, [MyCols.CURRENT_DATA], init_state)
	save_as_pickle(df, pkl)


def update_enum_pickle(file_path, df, pkl, clear_enum_file=False):
	"""
	Print a list of variable names with corresponding columns names in pickle dataframe
	:param df:
	:param file_path:
	:param clear_enum_file: empty dv enums
	:param pkl:
	:return:
	"""
	# -- DbState initialisation
	if clear_enum_file:
		reset_enum_file(file_path)

	statusPkl = 'DbState/{}_state'.format(file_path)

	dfState = read_pickle(statusPkl)
	dfState = dfState[dfState[MyCols.PKL_NAME] != pkl]

	oldCols = get_unique_values(dfState, MyCols.COL_NAME)
	newCols = get_cols_alphabetically(df)
	addedCols = diff_2_lists(newCols, oldCols)

	if len(addedCols) > 0:
		# --
		dfNewState = DataFrame(columns=[MyCols.COL_NAME, MyCols.PKL_NAME])
		dfNewState[MyCols.COL_NAME] = addedCols
		dfNewState[MyCols.PKL_NAME] = pkl
		# save_as_pickle(dfNewState, statusPkl)

		# -- Update list of db variables
		dfState = dfState.append(dfNewState)
		dfState.reset_index(inplace=True, drop=True)
		dfState.loc[dfState[MyCols.COL_NAME] == MyCols.CURRENT_DATA, MyCols.PKL_NAME] = pkl
		save_as_pickle(dfState, statusPkl)

	cols = get_unique_values(dfState, MyCols.COL_NAME)

	# -- Print list dv fields
	#
	# for col in cols:
	#     if col == MyCols.CURRENT_DATA:
	#         print('{} = \'{}\''.format(col.strip().replace(' ', '_').lower(), pkl))
	#     else:
	#         print('{} = \'{}\''.format(col.strip().replace(' ', '_').lower(), col))

	# -- Write to file dv class
	print_debug('update {} class with pickle : {}'.format(file_path, pkl))
	write_enums_to_file(file_path, cols, pkl)
	save_as_pickle(df, pkl)


def write_enums_to_file(file_path, cols, cmt=''):
	"""
	Write enums to python file
	:param file_path: file name
	:param cols: enums of fields
	:param cmt: comment
	:return:
	"""
	with open(file_path + '.py', 'w+') as py_file:
		py_file.write('# ' + cmt + '\n')

		for col in cols:
			if col == MyCols.CURRENT_DATA:
				py_file.write('{} = \'{}\'\n'.format(col.strip().replace(' ', '_').lower(), cmt))
			else:
				py_file.write('{} = \'{}\'\n'.format(col.strip().replace(' ', '_').lower(), col))


def save_as_pickle(o, file_path: str):
	"""
	Save dataframe to pickle folder with fn name

	:param o: Object
	:param file_path: Filename
	"""

	folder_path, file_name = get_file_path_and_name(file_path)
	create_dir_if_exists_not('{}/{}'.format(MyFolders.PICKLES, folder_path))
	file_path = MyFolders.PICKLES + file_path + '.pickle'
	print_debug(file_path)
	pd.to_pickle(o, file_path)


def save_as_csv(df: pd.DataFrame, file_path: str):
	"""
	Save dataframe to excel folder with fn name
	:param df:
	:param file_path:
	:return:
	"""
	folder_path, file_name = get_file_path_and_name(file_path)
	create_dir_if_exists_not('{}/{}'.format(MyFolders.PICKLES, folder_path))
	file_path = MyFolders.EXCEL + file_path + '.csv'
	print_debug(file_path)
	df.to_csv(file_path, decimal=',')


def save_as_xlsx(df, file_path, nrows=0):
	if nrows > 0:
		df = df[:nrows]

	folder_path, file_name = get_file_path_and_name(file_path)
	create_dir_if_exists_not('{}/{}'.format(MyFolders.EXCEL, folder_path))

	filePath = MyFolders.EXCEL + file_path + '.xlsx'
	writer = pd.ExcelWriter(filePath, engine='xlsxwriter', date_format='dd/mmm/yyyy')

	# print df on sheet1
	df.to_excel(writer, 'Data')

	# print column names in alphabetical order on sheet 2
	df_cols = pd.DataFrame(df.columns)
	df_cols.to_excel(writer, 'Columns')

	print_check(filePath)
	writer.save()


def save_as_html(df, fn):
	file_path = MyFolders.HTML + fn + '.html'
	print_check(file_path)
	f = open(file_path, 'w')
	f.write(df.to_html(bold_rows=False, border=0))
	f.close()


def save_plot_as_png(fig, file_path, transparent=False, dpi=300, **kwargs):
	folder_path, file_name = get_file_path_and_name(file_path)
	create_dir_if_exists_not('{}/{}'.format(MyFolders.GRAPHS, folder_path))

	file_path = MyFolders.GRAPHS + file_path + '.png'
	print_debug(file_path)
	fig.savefig(file_path, format='png', transparent=transparent, dpi=dpi, **kwargs)


def read_pickle(fn: str) -> DataFrame:
	"""
	Create a dataframe from pickle named 'fn'
	:rtype: object
	:param fn: Pickle file name
	:return: DataFrame
	"""
	return pd.read_pickle(MyFolders.PICKLES + fn + '.pickle')


# endregion


# region STRINGS
# ------------------------------------------------------------------------------------------
def as_percent(v, precision='0.2'):
	"""Convert number to percentage string."""
	if isinstance(v, Number):
		return "{{:{}%}}".format(precision).format(v)
	else:
		raise TypeError("Numeric type required")


def as_no_decimal(v, precision='0.2'):
	"""Convert number to percentage string."""
	if isinstance(v, Number):
		return "{{:{}}}".format(precision).format(v)
	# return "{{:{}}}".format(precision).format(v)
	else:
		raise TypeError("Numeric type required")


def format_cols_as_pct(df):
	cols = get_numerical_cols(df)
	for col in cols:
		df[col] = df[col].apply(as_percent)

	return df


def color_negative_red(val):
	"""	Takes a scalar and returns a string with the css property `'color: red'` for negative strings, black otherwise."""
	color = 'red' if val < 0 else 'black'
	return 'color: %s' % color


def color_below_100_pct_red(val):
	"""	Takes a scalar and returns a string with the css property `'color: red'` for negative strings, black otherwise."""
	color = 'red' if val < 1 else 'black'
	return 'color: %s' % color


def left(s, pos):
	return s[:pos]


def right(s, pos):
	return s[-pos:]


def mid(s, offset, pos):
	return s[offset:offset + pos]


def is_number(n):
	return isinstance(n, (int, float, complex))


def sanitize_string(oldStr: str, make_upper=True) -> str:
	"""
	Returns a sanitized string: unicoded, with parenthesis, commas or apostrophes replaced by underscore
	:param make_upper: boolean
	:param oldStr:
	:return:
	"""

	newStr = unidecode(oldStr).strip()
	newStr = re.sub(r'\([^()]*\)', '', newStr)
	newStr = re.sub(r'[\' /,-]', '_', newStr)
	newStr = re.sub(r'_+', '_', newStr)
	newStr = newStr.strip('_')

	if make_upper:
		newStr = newStr.upper()

	# --
	# print_debug('\'{}\' converted to \'{}\''.format(oldStr, newStr))
	return newStr


def get_french_translation(str) -> str:
	"""
	Returns from translation from MyDfs.FrenchDict
	:param str:
	:return:
	"""
	if str in MyDfs.FrenchDict.keys():
		return MyDfs.FrenchDict[str]
	else:
		return str


# endregion


# region COLUMNS STATS
# ------------------------------------------------------------------------------------------
def get_cols_with_value(df: DataFrame, values: list):
	cols = []

	for col in df.columns:
		if (df[col].isin(values)).any():
			cols.append(col)

	print_debug(cols)

	return cols


def count_unique_categories_per_cols(df: DataFrame):
	"""
		Decide which categorical variables you want to use in model
		Count how many actual categories are represented in each of the dataframe columns

		:type df:DataFrame
	"""
	# TODO : format string
	for col in df.columns:
		if df[col].dtypes == 'object':
			uniqueCats = len(df[col].unique())
			print("Feature '{col}' : {uniqueCats} unique categories".format(col=col, uniqueCats=uniqueCats))


def get_col_unique_counts(df, col):
	uniqueCats = df[col].unique().tolist()
	if np.nan in uniqueCats:
		uniqueCats = [x for x in uniqueCats if x is not np.NaN]
		uniqueCats.append('N/A')
	# -- print(type(examples))

	uniqueCats = sorted(uniqueCats)
	print_debug('COL: {}'.format(col), ' Nb of CAT: {}'.format(len(uniqueCats)), ' EX : {}'.format(uniqueCats[:100]))
	return uniqueCats


def get_cols_unique_counts(df, cols):
	cols.sort()
	# print(cols, df)
	for col in cols:
		get_col_unique_counts(df, col)


def get_unique_values(df, col, with_null=False):
	"""
	Get unique values in column 'col' of dataframe 'df' (NaN values excluded)
	:param with_null:
	:param col: string
	:param df: DataFrame
	:return: list
	"""
	if with_null:
		return sorted(df[col].unique().tolist())

	return sorted(df[col].dropna().unique().tolist())


def drop_columns(df, cols_to_drop: list, errors='raise') -> DataFrame:
	"""
	Returns a dataframe with cols_to_drop dropped
	:param errors : {‘ignore’, ‘raise’}, default ‘raise’ - If ‘ignore’, suppress error and only existing labels are dropped.
	:param cols_to_drop: list of columns to drop
	:param df: initial dataframe
	:return df: dataframe with dropped columns
	"""
	# print_debug("Dropped columns :", cols_to_drop)
	return df.drop(cols_to_drop, axis=1, errors=errors)


def rename_column_to(df, old_name, new_name, level=None):
	"""
	rename column 'old_name' by 'new_name" in place
	:param df: DataFrame
	:param old_name: str
	:param new_name: str
	:return:
	"""
	df.rename(columns={old_name: new_name}, inplace=True, level=level)


def replace_values_in_col_w_mean(df: DataFrame, col: str, vals: list) -> DataFrame:
	newVal = df.loc[~df[col].isin(vals), col].mean()
	new_col = '{}_{}'.format(col, MyCols.MEAN)
	print_debug(col, newVal)
	df[new_col] = df[col]
	df.loc[df[col].isin(vals), new_col] = newVal

	return df


def replace_values_in_col_w_median(df: DataFrame, col: str, vals: list) -> DataFrame:
	# col, vals, df = (wka.ord_education, [-1], dfX)
	newVal = df.loc[~df[col].isin(vals), col].median()
	new_col = '{}_{}'.format(col, MyCols.MEDIAN)
	print_debug(col, newVal)
	df[new_col] = df[col]
	df.loc[df[col].isin(vals), new_col] = newVal


def recode_col_to_num(df, col_to_encode, col):
	le = LabelEncoder()
	le.fit(df[col_to_encode].values)
	df[col] = le.transform(df[col_to_encode])


def get_cols(df):
	return list(df.columns)


def get_col_count(col, nb_rows='All'):
	if nb_rows == 'All':
		this_count = col.value_counts().sort_values(ascending=False)
	else:
		this_count = col.value_counts().sort_values(ascending=False).head(nb_rows)
	print(this_count, len(this_count))


def get_cols_with_negative_values(df: DataFrame) -> list:
	"""
	Return list of columns in DataFrame 'df' with negative values
	:param df:
	:return:
	"""
	numCols = get_numerical_cols(df)
	return df[numCols].columns[(df[numCols] < 0).any()].tolist()


def get_cols_with_null_values(df: DataFrame) -> list:
	"""
	Returns list of columns with null values in DataFrame 'df'
	:param cols:
	:return: list
	"""
	dfCols = DataFrame(get_nulls_count_per_cols(df))
	dfCols.columns = [MyCols.COUNT]
	return dfCols[dfCols[MyCols.COUNT] > 0].index.tolist()


def get_nulls_count_per_cols(df):
	dfCount = df.isnull().sum().sort_values(ascending=False)
	return dfCount


def get_count_per_cols(df):
	df_count = df.count().sort_values(ascending=False)
	return df_count


# -- TO REMOVE - 27.05.2018: after refactoring
# -------------------------------------------------------------------------------------- OLD
# def dummy_df(df: DataFrame, cols: list) -> DataFrame:
#     """Dummies all the categorical variables used for modeling from the column list"""
#     nbOfCols = len(df.columns)
#     print('{} columns in DataFrame'.format(nbOfCols))
#     print('{} columns to dummy'.format(len(cols)))
#
#     for col in cols:
#         print_debug("Dummied column : {}".format(col))
#         dummies = pd.get_dummies(df[col], prefix='DUM_' + col, dummy_na=False)
#         # df = df.drop(col, 1)
#
#         df = pd.concat([df, dummies], axis=1)
#
#     print_debug("{} columns added in DataFrame".format(len(df.columns) - nbOfCols), 2)
#
#     return df
# -------------------------------------------------------------------------------------- NEW
def dummy_df(df: DataFrame, cols: list, prefix='DUM', dummy_na=False) -> DataFrame:
	"""Dummies all the categorical variables used for modeling from the column list"""
	nbOfCols = len(df.columns)
	dfDummies = df[cols]
	if len(prefix) > 0:
		dfDummies.columns = ['{}_{}'.format(prefix, x) for x in cols]

	dfDummies = pd.get_dummies(dfDummies, dummy_na=dummy_na)
	dfDummies = df.join(dfDummies)

	dfDummies.columns = [sanitize_string(x) for x in dfDummies.columns]
	print_debug("{} columns added in DataFrame".format(len(dfDummies.columns) - nbOfCols), 2)

	return dfDummies


def add_interaction_columns(df, cols, degree=2):
	poly = PolynomialFeatures(degree)
	dfInteractions = DataFrame(poly.fit_transform(df[cols]))
	dfInteractions.index = df.index
	# interaction_cols = poly.get_feature_names(df[cols].columns)
	# interaction_cols = [x.replace(' ', '__').replace('^', '_POW') for x in interaction_cols]

	# # -- Rename columns
	# dfInteractions = dfInteractions.ix[:, 1:]
	# dfInteractions.columns = interaction_cols[1:]
	dfInteractions.columns = ['NUM_{}'.format(x) for x in np.arange(0, len(dfInteractions.columns))]

	df = drop_columns(df, cols)
	df = pd.concat([df, dfInteractions], axis=1)
	return df


def add_power_col(df, cols, pwr):
	"""
	Add numerical columns raised to the power of 'pwr'
	:param df: DataFrame
	:param cols: List of columns
	:param pwr: Power
	:return:
	"""
	cols_num = get_numerical_cols(df)

	for col in cols:
		if col in cols_num:
			col_power = '{}_PWR_{}'.format(col, pwr)
			df[col_power] = df[col] ** pwr
	# print(MyPrint.INFO + 'col: ' + col_power)


def add_log_col(df, cols):
	cols_num = get_numerical_cols(df)

	for col in cols:
		if col in cols_num:
			col_power = '{}_LOG'.format(col, )
			df[col_power] = np.log(df[col])
			print(MyPrint.INFO + 'add col ' + col_power)

			col_power = '{}_LOG2'.format(col)
			df[col_power] = np.log2(df[col])
			print(MyPrint.INFO + col_power)

			col_power = '{}_LOG10'.format(col)
			df[col_power] = np.log10(df[col])
			print(MyPrint.INFO + col_power)


def get_cols_with_nans(df):
	return df.columns[pd.isnull(df).any()].tolist()


def get_non_numerical_cols(df):
	"""Get list of non numerical columns in the dataframe"""
	non_numerical_cols = []

	for col_name in df.columns:
		if df[col_name].dtypes not in ['uint8', 'uint32', 'uint64', 'int8', 'int32', 'int64', 'float16', 'float32',
									   'float64']:
			non_numerical_cols.append(col_name)

	return non_numerical_cols


def is_serie_numerical(ser: pd.Series):
	return (ser.dtypes in ['uint8',
						   'uint32',
						   'uint64',
						   'int8',
						   'int32',
						   'int64',
						   'float16',
						   'float32',
						   'float64'])


def get_date_cols(df):
	date_cols = []

	for col in df.columns:
		if df[col].dtypes == 'datetime64[ns]':
			date_cols.append(col)

	return sorted(date_cols)


def get_numerical_cols(df):
	"""Get list of non numerical columns in the dataframe"""
	numerical_cols = []

	for col_name in df.columns:
		if is_serie_numerical(df[col_name]):
			numerical_cols.append(col_name)
		else:
			newCol = pd.to_numeric(df[col_name], errors='coerce')
			# print_debug(col_name)

			if newCol.isnull().sum() == df[col_name].isnull().sum() & is_serie_numerical(newCol):
				numerical_cols.append(col_name)

	return sorted(numerical_cols)


def get_stats(group):
	return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}


def grouped_by(cols, df):
	return df[cols].groupby(cols)


def get_grouped_size(cols, df):
	df = grouped_by(cols, df).size()
	return df


def get_unique_rows(df: pd.DataFrame, cols: list, with_total=False):
	"""
	Returns the total numbers of rows with unique values in passed columns list
	:param cols: list of columns
	:param df:
	:param with_total: Display
	:return:
	"""

	len_df = len(df)
	df = df[cols].groupby(cols).size().reset_index()

	df.columns = cols + [MyCols.TMP]
	totalValue = df[MyCols.TMP].sum()

	cnt_nulls = len_df - totalValue

	if cnt_nulls > 0:
		df.loc[MyCols.NULL_VALUES] = cnt_nulls

	if with_total:
		df.loc[MyCols.TOTAL] = ''
		df[MyCols.TMP][MyCols.TOTAL] = totalValue
		df[MyCols.PERCENT] = (df[MyCols.TMP] / totalValue).astype(np.float).round(2)
		df[MyCols.CUM_SUM] = df[MyCols.PERCENT].cumsum()
		df.loc[MyCols.TOTAL, MyCols.CUM_SUM] = ""

		df[MyCols.TOTAL] = df[MyCols.TMP]
		df = drop_columns(df, [MyCols.TMP])

		return df, totalValue

	else:
		df.columns = cols + [MyCols.TOTAL]
		return df


def get_unique_rows_by_column(df, with_total=False):
	for col in get_cols_alphabetically(df):
		dfTmp = get_unique_rows(df, [col], with_total)
		print_debug(dfTmp)


def get_values_count_by_column(df, filter_values):
	cols = get_cols_with_value(df, filter_values)
	df[cols] = df[cols].applymap(lambda x: sanitize_string(x))
	filter_values = [sanitize_string(x) for x in filter_values]
	print_full(get_unique_rows(df, cols).head())
	# -- Rotate columns, get counts
	df = df[cols].stack().reset_index().ix[:, 1:]
	# print_full(df.head(30))
	df.columns = [MyCols.COL_NAME, MyCols.VALUE]
	df = pd.pivot_table(df, index=MyCols.VALUE, columns=[MyCols.COL_NAME], aggfunc=len)
	df = df.loc[df.index.isin(filter_values)].T
	return df


# endregion


# region COLUMNS NAMING MANIPULATION
# ------------------------------------------------------------------------------------------


def get_cols_alphabetically(df):
	txtCols = ['{}'.format(col) if (isinstance(col, (int, float, complex))) else col for col in df.columns]
	return sorted(txtCols)


def reorder_column_to(df, col, pos):
	cols = list(df.columns)
	cols.insert(pos, cols.pop(cols.index(col)))
	df = df[cols]
	return df


def move_column_to_first(df, col):
	return reorder_column_to(df, col, 0)


def move_column_to_last(df, col):
	return reorder_column_to(df, col, len(df.columns) - 1)


def move_column_after(df, col_to_move, col_pos):
	cols = list(df.columns)
	pos = cols.index(col_pos)
	return reorder_column_to(df, col_to_move, pos + 1)


def move_column_before(df, col_to_move, col_pos):
	cols = list(df.columns)
	pos = cols.index(col_pos)
	return reorder_column_to(df, col_to_move, pos)


#
# def replace_values_in_col(df, col, new_values, new_col='TMP'):
#
#     old_values = get_unique_values(col, df)
#     diff_2_lists(old_values, new_values)
#
#     n = len(old_values) - len(new_values)
#
#     if n == 0:
#         dfTmp = DataFrame({col: old_values, new_col: new_values})
#         df = pd.merge(df, dfTmp, how='left', on=col)
#         print_debug(get_unique_rows([col, new_col], df))
#         print_debug(new_col)
#         if new_col == 'TMP':
#             df = drop_columns([col], df)
#             replace_in_col_names(df, new_col, col)
#     else:
#         print_warn('No changes where made: {} values missing.'.format(n))
#
#     return df


def replace_str_in_cols(df: pd.DataFrame, cols: list, old_str: str, new_str: str):
	"""
	Replace in 'df' DataFrame columns 'cols', string 'old_str' with string 'new_str' in place
	:param df: DataFrame
	:param cols: list of columns in which renaming will occur
	:param old_str: str
	:param new_str: str
	:return:
	"""
	for col in cols:
		df[col] = df[col].str.replace(old_str, new_str).str.strip()


def replace_in_col_names(df: DataFrame, old_str, new_str, level=None):
	"""
	Rename in column names old_str with new_str in place
	:param df: DataFrame
	:param old_str: str
	:param new_str: str
	:param level: int or level name
	:return:
	"""

	if level is not None:
		cols = df.columns.get_level_values(level).tolist()
	else:
		cols = df.columns

	print_debug('Renamed columns :')
	for col in cols:
		print_debug(col, old_str)
		if old_str in col:
			newCol = col.replace(old_str, new_str)
			df.rename(columns={col: newCol}, inplace=True, level=level)
			print('{} to {}'.format(col, newCol))


def col_prefix_to_suffix(df, prefix, suffix):
	for col in df.columns:
		# print(col, prefix, suffix)

		if left(col, len(prefix)) == prefix:
			newCol = right(col, len(col) - len(prefix))
			newCol += suffix
			rename_column_to(df, col, newCol)
			print(col, newCol)


def col_add_suffix(df, prefix, suffix):
	for col in df.columns:
		# print(col, prefix, suffix)

		if left(col, len(prefix)) == prefix:
			newCol = col + suffix
			rename_column_to(df, col, newCol)
			print(col, newCol)


def col_add_prefix(df, prefix, new_prefix):
	for col in df.columns:
		# print(col, prefix, suffix)

		if left(col, len(prefix)) == prefix:
			newCol = new_prefix + col
			rename_column_to(df, col, newCol)
			print(col, newCol)


def get_cols_with(df: pd.DataFrame, s: str, match_case=False):
	return get_items_with(get_cols_alphabetically(df), s, match_case)


def get_list_index_by_rank(items, reverse=False):
	seq = sorted(items, reverse=reverse)
	indexes = [seq.index(i) for i in items]
	return indexes


def get_items_with(lst, s: str, match_case=False):
	if not match_case:
		lst = [x.upper() for x in lst]
		s = s.upper()

	return [x for x in lst if s in x]


def get_cols_with_prefix(df, prefix) -> list:
	return get_items_with_prefix(df.columns, prefix)


def get_cols_with_prefixes(df: DataFrame, prefixes: list) -> list:
	all_cols = []
	for p in prefixes:
		all_cols += get_cols_with_prefix(df, p)

	return sorted(list(set(all_cols)))


def get_items_with_prefix(list, prefix):
	return [item for item in list if left(item, len(prefix)) == prefix]


def get_cols_with_suffix(df, suffix):
	return get_items_with_suffix(df.columns, suffix)


def get_items_with_suffix(list, suffix):
	return [item for item in list if right(item, len(suffix)) == suffix]


def get_cols_with_prefix_suffix(df, prefix, suffix):
	return get_items_with_suffix(get_cols_with_prefix(df, prefix), suffix)


def set_cols_to_str(df):
	col_names = ['{}'.format(col) for col in df.columns]
	df.columns = col_names
	return df


def diff_2_lists(l1, l2):
	return list(set(l1) - set(l2))


def common_2_lists(l1, l2):
	return list(set([x for x in l1 if x in l2]))


def diff_2_df_columns(df1, df2):
	return diff_2_lists(get_cols(df1), get_cols(df2))


def common_2_df_columns(df1, df2):
	return common_2_lists(get_cols(df1), get_cols(df2))


def list_duplicate_columns(df):
	originalCols = get_cols_alphabetically(df)
	duplicateCols = list(set([x for x in originalCols if originalCols.count(x) > 1]))
	print(MyPrint.INFO + 'Duplicate columns : {}'.format(duplicateCols))
	return duplicateCols


def remove_col_with_prefix(df, prefix):
	cols = get_cols_with_prefix(df, prefix)

	print(prefix, cols)
	return drop_columns(df, cols)


def remove_col_with_suffix(df, suffix):
	cols = get_cols_with_suffix(df, suffix)

	print(suffix, cols)
	return drop_columns(df, cols)


def reorder_col_alphabetically(df):
	return df[sorted(df.columns)]


def merge_corr_cols(df, merging_cols: list, new_col, remove_old=True):
	df[MyCols.TMP] = 0
	df.loc[(df[merging_cols] == 1).any(axis=1), MyCols.TMP] = 1

	if remove_old:
		df.drop(merging_cols, axis=1, inplace=True)
		print_info("Drop columns {} & after merging into {}".format(", ".join(merging_cols), new_col))
	elif new_col in merging_cols:
		new_col = 'NEW_' + new_col
		print_info("Merged columns {} into {}".format(", ".join(merging_cols), new_col))

	rename_column_to(df, MyCols.TMP, new_col)


def merge_invcorr_cols(df, kept_col, merged_col, new_col=''):
	df.drop(merged_col, axis=1, inplace=True)
	if len(new_col) > 0:
		rename_column_to(df, kept_col, new_col)
		print("INFO: Merge inversely correlated columns {} into {}".format(merged_col, new_col))
	else:
		print("INFO: Merge inversely correlated columns {} into {}".format(merged_col, kept_col))


def sanitize_col(col, df):
	oldValues = get_unique_values(df, col)
	newValues = [sanitize_string(x) for x in oldValues]

	valDict = dict(zip(oldValues, newValues))
	print_debug(valDict)

	for key, value in valDict.items():
		replace_str_in_cols(df, [col], key, value)


# endregion


# region COLUMNS MANIPULATION
# ------------------------------------------------------------------------------------------

def scale_min_max_df(df):
	sc = MinMaxScaler()
	df_scaled = DataFrame(sc.fit_transform(df.astype(float)), columns=df.columns)
	df_scaled.index = df.index
	return df_scaled


# REMOVE 14102018
# def select_from_index(df, ix):
#     df = df[df.index.isin(ix)]
#     return df
# END REMOVE

# REMOVE : 19092018
# def fillna_in_cols_with_null_values(df):
#     nullCols = get_cols_with_null_values(df)
#     print_debug('Cols with null values:', nullCols)
#     df[nullCols] = df[nullCols].fillna(0)
# END REMOVE

def remove_values(df: DataFrame, vals: list, col: str):
	"""
	Returns a new DataFrame with values listed in 'lst' from column 'col'
	:param vals:List
	:param col: str
	:param df: DataFrame
	:return:
	"""
	# CHECK

	cnt = len(df)
	initial_cnt = len(df)

	for v in vals:
		df = df[df[col] != v]

		removed_cnt = cnt - len(df)
		cnt = len(df)
		print_debug('Removed from \'{}\' {} values of type \'{}\' = {}'.format(
			col,
			removed_cnt,
			v,
			as_percent(removed_cnt / initial_cnt)))

		removed_total = initial_cnt - cnt

		print_debug('Total removed from \'{}\' {} out of {} ({}) values (remains: {})'.format(
			col,
			removed_total,
			initial_cnt,
			as_percent(removed_total / initial_cnt),
			cnt
		))

	return df


def find_outliers_tukey(x):
	q1 = np.percentile(x, 25)
	q3 = np.percentile(x, 75)
	iqr = q3 - q1
	floor = q1 - 1.5 * iqr
	ceiling = q3 + 1.5 * iqr
	outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
	outlier_values = list(x[outlier_indices])

	return outlier_indices, outlier_values


def remove_tukey_outliers(df, col):
	"""
	Return from DataFrame 'df' a new DataFrame without tukey outliers based on colunm 'col' outliers
	:param df:
	:param col:
	:return:
	"""
	initial_cnt = len(df)

	outliers_ix, outliers_val = find_outliers_tukey(df[col])
	if len(outliers_ix) > 0:
		df = df.drop(outliers_ix)

	cnt = len(df)
	removed_total = initial_cnt - cnt

	print_debug('Total removed from \'{}\' {} out of {} values ({}, remains: {})'.format(
		col,
		removed_total,
		as_percent(removed_total / initial_cnt),
		initial_cnt,
		cnt
	))

	print_debug('Removed the following values: ', outliers_val)
	return df


def get_interaction_cols(df):
	return list(combinations(list(df.columns), 2))


def remove_all_0_1_cols(df):
	df = df.drop(get_cols_where_all(df), axis=1)
	df = df.drop(get_cols_where_all(df, 1), axis=1)
	return df


def get_cols_where_all(df, v=0):
	return df.columns[(df == v).all()]


def get_indexes_where_all(df, v=0):
	return df.index[(df == v).all()]


def add_interactions(df, old_str=None, new_str=None):
	startTime = time.time()
	print_duration('Start: {}'.format(startTime))

	# CHECK
	# df = dfInteractions

	# -- Get feature names
	combos = get_interaction_cols(df)
	addedCols = ['_&_'.join(x) for x in combos]

	if new_str is not None:
		addedCols = [x.replace(old_str, new_str) for x in addedCols]

	cols = df.columns.tolist()
	newCols = cols + addedCols
	print_debug("All columns :", newCols, len(newCols))

	# -- Find interactions
	poly = PolynomialFeatures(interaction_only=True, include_bias=False)
	dfNew = poly.fit_transform(df)
	dfNew = pd.DataFrame(dfNew)
	dfNew.columns = newCols

	# -- Remove interaction terms with all 0 or 1 values
	dfNew = remove_all_0_1_cols(dfNew)

	return dfNew


def add_cols_interactions(df: DataFrame, cols: list, degree=2, drop_cols=True, interaction_only=False, include_bias=True, old_str=None, new_str=None) -> DataFrame:
	dfInteractions = df[cols]

	# -- Find interactions
	poly = PolynomialFeatures(degree, interaction_only, include_bias)
	dfInteractions = DataFrame(poly.fit_transform(dfInteractions), columns=poly.get_feature_names(dfInteractions.columns))
	dfInteractions.index = df.index

	# -- Clean up column
	dfInteractions = drop_columns(dfInteractions, ['1'] + cols)
	dfInteractions = remove_all_0_1_cols(dfInteractions)
	dfInteractions.columns = [sanitize_string(x) for x in dfInteractions.columns]
	dfInteractions.columns = [x.replace('^', '_POW') for x in dfInteractions.columns]

	if drop_cols:
		df = drop_columns(df, cols)

	if new_str is not None:
		dfInteractions = [x.replace(old_str, new_str) for x in dfInteractions.columns]

	# print_debug(get_cols_alphabetically(dfInteractions))
	common_cols = common_2_lists(df, dfInteractions)

	if len(common_cols):
		print_warn('Common columns before adding interactions (): {}'.format(', '.join(common_cols)))

	# -- Concatenate back all columns
	df = pd.concat([df, dfInteractions], axis=1)

	return df


def get_xy_data(y_col, df, dummy_cols=[], values_to_remove=None):
	for col in df.columns:
		rows, total = get_unique_rows(df, [col])
		missing = len(df) - total
		print(col, ' | missing nan : ', missing)

		if missing > 0:
			print(MyPrint.WARNING + 'missing y value for x_y model in column > ' + col)
			break

	if len(dummy_cols) > 0:
		# get_cols_unique_counts(dummy_cols, df)
		df = dummy_df(df, dummy_cols)

	if values_to_remove:
		df = remove_values(df, values_to_remove, y_col)

	y = df[y_col]
	X = np.array(df.drop([y_col], 1))

	return X, y, df


def get_xy_cols(y_col, df):
	for col in df.columns:
		rows, total = get_unique_rows(df, [col])
		missing = len(df) - total
		print(col, ' | missing nan : ', missing)

		if missing > 0:
			print(MyPrint.WARNING + 'missing y value for x_y model in column > ' + col)
			break

	y = df[y_col]
	X = drop_columns(df, [y_col])
	cols = X.columns.tolist()

	return np.array(X), y, cols


def get_x_data(dummy_cols, df):
	for col in df.columns:
		rows, total = get_unique_rows(df, [col])
		missing = len(df) - total
		print(col, rows, ' | missing nan : ', missing)

		if missing > 0:
			print(MyPrint.WARNING + 'missing y value for x_y model in column > ' + col)
			break

	if len(dummy_cols) > 0:
		# get_cols_unique_counts(dummy_cols, df)
		df = dummy_df(df, dummy_cols)

	return df


# endregion


# region TIME
duration = {}


class TASK:
	start = 'START'
	end = 'END'


def print_time(task_name=TASK.start):
	fulltime = time.gmtime()
	duration[task_name] = fulltime
	print(task_name, time.strftime("%a, %d %b %Y %H:%M:%S", fulltime))


def print_duration(task_name=TASK.start, spaced=1):
	print('\n' * spaced, MyPrint.DURATION, '\n',
		  task_name,
		  'ends after'.format(TASK.end), timedelta(seconds=time.mktime(time.gmtime()) - time.mktime(duration[task_name])))


# print_time()
# print_duration()

# endregion


# region CALCULATION

def round_to_nearest(x, base=5):
	return int(base * round(float(x) / base))


def add_bin_labels(df: pd.DataFrame, col: str = 'NUM_AGE', label_col: str = 'AGE_GROUP', ord_col: str = 'ORD_AGE_GROUP', bin_size: int = 5, min_cut: int = 0, max_cut: int = 100,
				   has_fuzzy_lowest: bool = False, has_fuzzy_highest: bool = True) -> None:
	"""

	:param df:
	:param min_cut: bucketing lower bound
	:param max_cut: bucketing higher bound
	:param col: column to bucket
	:param label_col: label column
	:param ord_col: bin numbering column
	:param bin_size: bin size
	:param has_fuzzy_lowest: if true buckets any value lower than min_cut
	:param has_fuzzy_highest: if true buckets any value greater than max_cut
	"""
	# Test settings
	# df = dfGenPop
	# min_cut = 15
	# max_cut = 65
	# col = 'NUM_AGE'
	# label_col = 'AGE_GROUP'
	# ord_col = 'ORD_AGE_GROUP'
	# bin_size = 5
	# has_fuzzy_lowest = True
	# has_fuzzy_highest = True
	# Test settings

	bins, bins_labels = get_bins(min_cut, max_cut, bin_size, has_fuzzy_highest, has_fuzzy_lowest)

	# -- Check bins are same size
	print_debug(bins)
	print_debug(bins_labels)
	print_debug(len(bins) - len(bins_labels))

	df[label_col] = pd.cut(df[col], bins=bins, labels=bins_labels)
	# df[label_col] = pd.cut(df[col], bins=bins, labels=bins_labels, include_lowest=has_fuzzy_lowest, right=has_fuzzy_highest)
	df[ord_col] = [bins_labels.index(x) + 1 for x in df[label_col]]

	print_debug(print_full(get_unique_rows(df, [col, label_col, ord_col])))


def get_bins(min_cut, max_cut, bin_size, has_fuzzy_highest=False, has_fuzzy_lowest=True):
	bins = np.arange(min_cut, max_cut + bin_size, bin_size).tolist()
	bins_labels = ['{} - {}'.format(x, x + (bin_size - 1)) for x in bins if x < max_cut]
	bins = [x - 1 for x in bins]
	# bins_labels[0] = '< 15'
	if has_fuzzy_lowest:
		bins_labels.insert(0, '< {}'.format(min_cut))
		bins.insert(0, float('-inf'))
	if has_fuzzy_highest:
		bins_labels.append('>= {}'.format(max_cut))
		bins.append(float('inf'))
	return bins, bins_labels


# region DAL


class DbDefault:
	# MYSQL
	server = 'self1014.selfserveur.net'
	driver = "{MySQL ODBC 5.3 ANSI Driver}"
	db = 'squashb_squashbe'
	user = "squashb"
	pwd = "squash2013"


def sql2df(sql, server=DbDefault.server, db=DbDefault.db, driver=DbDefault.driver, user=DbDefault.user,
		   pwd=DbDefault.pwd):
	conn_string = "DRIVER=" + driver + ";" + "SERVER={};DATABASE={};uid={};pwd={};".format(server, db, user, pwd)
	conn = pyodbc.connect(conn_string)
	cursor = conn.cursor()
	df = pd.read_sql(sql, conn)
	conn.close()

	# print('INFO: ' + conn_string)

	return df


def sql2df_from_conn(sql: str, conn_string: str):
	conn = pyodbc.connect(conn_string)
	cursor = conn.cursor()
	df = pd.read_sql(sql, conn)
	conn.close()

	print_info(conn_string)

	return df


def execute_sql(sql, server=DbDefault.server, db=DbDefault.db, driver=DbDefault.driver, user=DbDefault.user,
				pwd=DbDefault.pwd):
	print("From db: {} on {}".format(db, server))
	conn_string = "DRIVER=" + driver + ";" + "SERVER={};DATABASE={};uid={};pwd={};".format(server, db, user, pwd)
	conn = pyodbc.connect(conn_string)
	cursor = conn.cursor()
	cursor.execute(sql)
	conn.close()

# endregion
