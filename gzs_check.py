import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime

np.seterr(divide='ignore', invalid='ignore')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# 非监督
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 计算 轮廓系数，CH 指标，DBI
from sklearn.cluster import DBSCAN

### 片段动作

# 单人坐骑
ZQ_column = ['LOG_HORSEBOOK_POP', 'LOG_HORSEBOOK_PUSH', 'LOG_HORSE_LIFE_JINGPO', 'LOG_HORSE_LEARN_GOLDEN_SKILL',
             'LOG_HORSE_REROLL_GOLDEN_SKILL', 'LOG_HORSE_COMFIRM_REROLL', 'LOG_HORSE_DELETE_GOLDEN_SKILL',
             'LOG_HORSE_UPGRADE_GOLDEN_SKILL']
# 天工匣
TGX_column = ['LOG_TYPE_EQUIP_REFINE', 'LOG_TYPE_EQUIP_REFINE_RESULT']
# 锻造
DZ_column = ['LOG_FORGE', 'LOG_PLAYER_EQUIP_FORGE_INFO', 'LOG_FORGE_COST', 'GAME_LOG_FORGE_START']
# 法宝
FB_column = ['LOG_FABAO_REFINE', 'LOG_FABAO_FUSION', 'LOG_CORE_SKILL_UP', 'LOG_CORE_SKILL_RESET',
             'LOG_FASHION_PANEL_JIHUO', 'LOG_FASHION_PANEL_RESOLVE']
# 双人坐骑
SRZQ_column = ['LOG_HORSE_BLESS,LOG_HORSE_UPLEV,LOG_HORSE_SKILL_RESET,LOG_HORSE_TIANFU']
# 商店
SD_column = []
# 外观
WG_column = ['LOG_FASHION_JIHUO', 'LOG_SURFACE_POINT']
# 神器
SQ_column = ['LOG_EMPEROR_FUSION', 'LOG_EMPEROR_UP', 'LOG_EMPERORSHIPSKILL_ACTIVATE', 'LOG_EMPERORSHIPSKILL_LEVELUP',
             'LOG_EMPERORSHIPSKILL_WASH', 'LOG_EMPERORSHIPSKILL_WASH_OPER']
# 元神
YS_column = ['LOG_SOUL_SKILL_CHANGE', 'LOG_SOUL_ADD_EXP', 'LOG_SOUL_EXCHANGE', 'LOG_SOUL_BUY_EXP',
             'LOG_TYPE_YUANSHENLI_ADD', 'LOG_TYPE_YUANSHENLI_SUB']
# 仙侣
XL_column = ['LOG_LOVE_REPAY_ITEM_MAIL', 'LOG_LOVE_MARRY_CHANGE', 'LOG_LOVE_PARADE_AWARD', 'LOG_LOVE_PARADE_COST',
             'LOG_LOVE_PARADE_SWEETS', 'LOG_LOVE_PARADE_FORCE_END', 'LOG_LOVE_CHALLENGE_MIN_AWARD',
             'LOG_LOVE_CHALLENGE_KILL', 'LOG_LOVE_CHALLENGE_ENERGY', 'LOG_LOVE_CHALLENGE_WIN_LOSE',
             'LOG_DRAW_LINE_OPEN', 'LOG_DRAW_LINE_ADVANCE_CLOSE', 'LOG_DRAW_LINE_PAY', 'LOG_DRAW_LINE_AWARD',
             'LOG_DRAW_LINE_BARRIER', 'LOG_DRAW_LINE_SETTLE', 'LOG_DRAW_LINE_CHANGE_STATUS',
             'LOG_DRAW_LINE_SWEEP_AWARD', 'LOG_DRAW_LINE_SWEEP_MGR']
# 排行榜
PHB_column = []
# 帮会
BH_column = ['LOG_GUILD_CREATE_CONSUME', 'LOG_GUILD_CREATE', 'LOG_GUILD_LEV_UP', 'LOG_TYPE_GUILD_ADD_WORSHIP',
             'LOG_TYPE_GUILD_SUB_WORSHIP', 'LOG_TYPE_GUILD_GET_CHEST', 'LOG_TYPE_GUILD_ACCUSE',
             'LOG_TYPE_GUILD_CHANGE_NAME', 'LOG_TYPE_GUILD_OPEN_TREE_ANSWER_QUESTION',
             'LOG_TYPE_GUILD_OPEN_TREE_ACTIVITY', 'LOG_TYPE_GUILD_WATCH_STAR', 'LOG_TYPE_GUILD_FINISH_STONE_TASK',
             'LOG_TYPE_GUILD_HELP_FINISH_STONE_TASK', 'LOG_TYPE_GUILD_DONATE', 'LOG_TYPE_GUILD_OPEN_TREE_DRINK',
             'LOG_TYPE_GUILD_BOSS_REWARD', 'LOG_TYPE_GUILD_DICE_REWARD', 'LOG_TYPE_GUILD_TRIAL_JIHUO',
             'LOG_TYPE_GUILD_TRIAL_OPEN', 'LOG_TYPE_GUILD_TRIAL_JOIN', 'LOG_TYPE_GUILD_TRIAL_RESULT',
             'LOG_TYPE_GUILD_TRIAL_PLAYER_RESULT', 'LOG_TYPE_GUILD_TRIAL_JIHUO_SUCCESS',
             'LOG_TYPE_GUILD_TRIAL_OPEN_SUCCESS', 'LOG_CITY_WAR_GUILD', 'LOG_AUCTION_BID_GUILD_ITEM',
             'LOG_AUCTION_BID_GUILD_ITEM_FAILED', 'LOG_AUCTION_BID_GUILD_ITEM_SUCCESS',
             'LOG_AUCTION_BID_GUILD_ITEM_FAILED_MAIL', 'LOG_AUCTION_BID_GUILD_ITEM_SUCCESS_MAIL',
             'LOG_AUCTION_GUILD_SELL_ITEM', 'LOG_AUCTION_GUILD_BID_ITEM_SUCCESS', 'LOG_AUCTION_GUILD_BID_ITEM_BACK',
             'LOG_AUCTION_GUILD_ITEM_DELAY', 'LOG_TYPE_GUILD_PVP_GUILD_AWARD', 'LOG_GUILD_ALLOT_BONUS_CREATE',
             'LOG_TYPE_GUILD_PVP_BOSS_DIE_MGR', 'LOG_GUILD_AUTO_LEAVE', 'LOG_GUILD_AUTO_DISBAND',
             'LOG_CITY_WAR_WIN_GUILD', 'LOG_GUILD_SEND_ITEMS', 'LOG_GUILD_RECEIVE_ITEMS', 'LOG_GUILD_TIANZHU_ITEM_USE',
             'LOG_GUILD_TIANZHU_SUCC', 'LOG_GUILD_TIANZHU_REPAY_MAIL', 'LOG_GUILD_TIANZHU_PUNISH',
             'LOG_GUILD_TIANZHU_AWARD', 'LOG_GUILD_TIANZHU_RELIFE', 'LOG_GUILD_WATER_DO', 'LOG_GUILD_WATER_FRUIT_NEW',
             'LOG_GUILD_WATER_FRUIT_DO', 'LOG_GUILD_WATER_ACT_END', 'LOG_ADD_BANGGONG', 'LOG_GUILD_POSITION_AWARD',
             'LOG_GUILD_DONATION', 'LOG_ACT_HAOBANGZHU_VOTE', 'LOG_ACT_HAOBANGZHU_ITEM', 'LOG_ACT_HAOBANGZHU_PAY',
             'LOG_GUILD_BUILD_SUB_ITEM', 'LOG_GUILD_PAIR_PUBLISH', 'LOG_GUILD_PAIR_BACKOUT', 'LOG_GUILD_PAIR_MATCH',
             'LOG_GUILD_PAIR_SHARE', 'LOG_GUILD_PAIR_LEAVE', 'LOG_GUILD_WATER_DO_NEW', 'LOG_PROTECT_TREE_SETTLE_AWARD',
             'LOG_PROTECT_TREE_ADD_BUFF', 'LOG_PROTECT_TREE_WAVE_STRENGTH', 'LOG_PROTECT_TREE_WAVE_COSTTIME',
             'LOG_PROTECT_TREE_OPEN', 'LOG_CITY_WAR_UPDATE_HOLD_TIME', 'LOG_CITY_WAR_DELETE_HOLD_TIME',
             'LOG_CITY_WAR_END_GUILD_HOLD_TIME', 'LOG_CITY_WAR_AWARD_SEND_AWARD', 'LOG_CITY_WAR_AWARD_PLY_SEND_AWARD',
             'LOG_GUILD_GM_TRY_JOIN_GUILD']
# 聚灵台
JLT_column = ['LOG_GEM_LVL_UP', 'LOG_GEM_LVL_BREAK', 'LOG_GEM_LVL_UP_ALL', 'LOG_BAGGEM_TRANS', 'LOG_GEM_DRESS',
              'LOG_GEM_UNLOCK', 'LOG_GEM_UNDRESS', 'LOG_BAGGEM_LEACH', 'LOG_BAGGEM_ADJUST', 'LOG_TYPE_GEM_INFO',
              'LOG_GEM_DRESS_NEW_PAGE']
# 镏金炉
LJL_column = ['LOG_TYPE_EQUIP_FLYUP']
# 八卦印
BGY_column = ['LOG_SUIT_DRESS']
# 境界
JJ_column = []
# 仙府
XF_column = ['LOG_HOME_USE_ITEM', 'LOG_HOME_BUILD_LEVEL_UP', 'LOG_HOME_BUY_FIX', 'LOG_HOME_HOUSE_TO_ITEM',
             'LOG_HOME_HOUSE_CHANGE', 'LOG_HOME_HOUSE_NAME_CHANGE', 'LOG_HOME_GROW_SUB', 'LOG_HOME_GROW_ADD',
             'LOG_HOME_GROW_AWARD', 'LOG_HOME_STORE_TAKE_OUT', 'LOG_HOME_STORE_TAKE_IN', 'LOG_HOME_GARNISH_JIHUO',
             'LOG_HOME_USE_HOUSE_ITEM', 'LOG_HOME_ADD_HOUSE_ITEM', 'LOG_HOME_GROW_DO', 'LOG_HOME_GROW_GET',
             'LOG_HOME_GROW_CLEAR', 'LOG_HOME_GARNISH_USE', 'LOG_HOME_HOUSE_ERR_BACK', 'LOG_HOME_PLAYER_IL_HOUSE_OWNER',
             'LOG_HOME_PLAYER_IL_HOUSE_INMATE', 'LOG_HOME_START_SELL', 'LOG_HOME_SELL_JOIN', 'LOG_HOME_RESOURCE',
             'LOG_HOME_STEAL_LAST_COUNT', 'LOG_HOME_ATTRIBUTE', 'LOG_HOME_CROP_CHANGE', 'LOG_HOME_CROP',
             'LOG_FAIRY_ROBBER_KILL', 'LOG_FAIRY_ROBBER_TRIGGER', 'LOG_FAIRY_BANQUET_OPEN', 'LOG_HOME_BQNQUET_CHANGE',
             'LOG_HOME_ROBBER_CHANGE', 'LOG_FAIRY_BANQUET_SUBRES', 'LOG_HOME_USE_RES_ITEM', 'LOG_HOME_RESET_HOUSE']
# 精炼
JL_column = ['LOG_SMELT_JL', 'LOG_SMELT_XZ', 'LOG_SMELT_CC', 'LOG_SMELT_FORGET', 'LOG_SMELT_FORGET_ONE',
             'LOG_SMELT_RESELECT']
# 生活技能
SHJN_column = ['LOG_LIFE_SKILL_FORGET', 'LOG_LIFE_SKILL_STUDY', 'LOG_LIFE_SKILL_USE_OK', 'LOG_LIFE_SKILL_FORMULA_JIHUO',
               'LOG_LIFE_SKILL_FORMULA_USE', 'LOG_BAG_FUMO', 'LOG_LIFE_SKILL_USE_BEGIN', 'LOG_LIFE_SKILL_USE_FAIL']

# 福利
FL_column = ['LOG_TYPE_FINDER_FIND', 'LOG_TYPE_FINDER_INFO', 'LOG_TYPE_FINDER_ACCEPT', 'LOG_TYPE_FINDER_LAST_COUNT']
# 交易行
JYH_column = ['LOG_TYPE_TRADE_SELL_SUBMONEY', 'LOG_TRADE_GET_MONEY', 'LOG_TRADE_ITEM_SUB_OK', 'LOG_TRADE_ITEM_SELL_OK',
              'LOG_TRADEFACE_RESULT', 'LOG_TRADE_ITEM_OVER_ITEM', 'LOG_TRADE_ITEM_OVER_MONEY',
              'LOG_TRADE_SELL_MONEY_OK', 'LOG_TRADE_AUTO_COOL', 'LOG_TRADE_CANCEL_BACK_MONEY',
              'LOG_TRADE_CANCEL_BACK_ITEM', 'LOG_TRADE_GET_JUDGE_ITEM', 'LOG_TRADE_GET_JUDGE_ITEM_INFO']
# 装备鉴定
JD_column = ['LOG_PLAYER_IDENTIFY_ITEM']
# 历练之路
LLZL_column = ['LOG_EXPERIENCE_BATTLE_AWARD', 'LOG_EXPERIENCE_WEEK_AWARD', 'LOG_EXPERIENCE_TASK_AWARD',
               'LOG_EXPERIENCE_TASK_FINISH']
# 野外首领
YWSL_column = ['LOG_TYPE_WORLDBOSS_PLAYER_AWARD', 'LOG_TYPE_WORLDBOSS_LAST_HURT']
# 观星台
GXT_column = []
# 仙灵
XL2_column = ['GAME_LOG_PET_FEED', 'GAME_LOG_PET_EVO', 'GAME_LOG_PETSKILL_LEARN', 'GAME_LOG_PETSKILL_UPLEV',
              'GAME_LOG_PETSKILL_DEL', 'GAME_LOG_PETSKILL_RESET', 'GAME_LOG_PET_SKIN_UNLOCK', 'GAME_LOG_PET_EXPITEM',
              'LOG_PET_ATTR_LVL_UP', 'LOG_PET_STAGE_LVL_UP', 'LOG_PET_STAGE_UNLOCK_SKILL', 'LOG_PET_STAGE_SKILL_LVL_UP',
              'LOG_PET_GROUP_LVL_UP', 'LOG_PET_GROUP_LVL_JIHUO', 'LOG_PET_GROUP_INVAILD_INFO']
# 仙器
XQ_column = ['LOG_XIANQI_TXF_CHANGE']
# 侠义协助
XYXZ_column = []
# 奇缘祈福
QYQF_column = []

### 连续动作

# 商城
# LOG_TYPE_MALL_SHOP_BUY_START ------> LOG_TYPE_MALL_SHOP_BUY_RESULT
SC_column = ['LOG_TYPE_MALL_SHOP_BUY_START', 'LOG_TYPE_MALL_SHOP_BUY_RESULT']
# 炼魔阵
# LOG_FB_ENTER -----> LOG_TYPE_LMZ ------> LOG_FB_SETTLE
LMZ_column = ['LOG_TYPE_LMZ']
# 封魔录
# LOG_FB_ENTER ----->LOG_COPY_FML
FML_column = ['LOG_COPY_FML']
# 天降彩珠
# LOG_TYPE_COLLECT_GIFT_BEGIN -----> LOG_TYPE_COLLECT_GIFT_END
TJCZ_column = ['LOG_TYPE_COLLECT_GIFT_BEGIN', 'CHILDLOG_LOG_SKYBOOK_ADD_TIMES_23', 'LOG_TYPE_COLLECT_GIFT_END']
# 追捕邪魔
# LOG_NEW_CATCH_DEVIL_START	 ----> LOG_NEW_CATCH_DEVIL_AWARD
ZBXM_column = ['LOG_NEW_CATCH_DEVIL_START', 'CHILDLOG_LOG_SKYBOOK_ADD_TIMES_86', 'LOG_NEW_CATCH_DEVIL_AWARD']
# 副本
# LOG_FB_ENTER ----> LOG_COPY_ENDINFO
FB_column = ['LOG_COPY_ENDINFO']
# 赛跑活动
# LOG_ACT_START ------> LOG_ACT_RUNNING_AWARD
SPHD_column = ['LOG_ACT_RUNNING_AWARD']
# 蟠桃园
# LOG_ACT_START ----> CHILDLOG_LOG_ACT_START_10 ----> LOG_PEACH_TIME_OUT
PTY_column = ['CHILDLOG_LOG_ACT_START_10']
# 血魔幻境

XMHJ_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_14']
# 怪物围剿
GWWJ_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_73']
# 保护苦行头陀
BHKXTT_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_7']
# 护送李英琼
HSLYQ_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_4']
# 圣焰战场
SYZC_column = ['LOG_MULTI_WAR_BEGIN', 'CHILDLOG_LOG_SKYBOOK_ADD_TIMES_13', 'LOG_MULTI_WAR_END']
# 超度游魂
CDYH_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_69']
# 仙门试炼
XMSL_column = ['GAME_LOG_XIANMEN_TRAILS_START', 'GAME_LOG_XIANMEN_TRAILS']
# 秘宝蒙尘
MBFC_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_5']

###无数据
# 玄石之界
XSZJ_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_22']
# 攻城战
GCZ_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_11']
# 乱斗场
LDC_column = ['CHILDLOG_LOG_SKYBOOK_ADD_TIMES_8']
# 凤凰令
FFL_column = []
# 野外首领
YWSL_column = []
# 龟兔赛跑
GTSP_column = []
# ALL
MAIN_column = ZQ_column + TGX_column + DZ_column + SRZQ_column + SD_column + WG_column + SQ_column + YS_column + XL_column + PHB_column + BH_column + JLT_column + LJL_column + BGY_column + JJ_column + XF_column + JL_column + SHJN_column + FL_column + JYH_column + JD_column + LLZL_column + YWSL_column + GXT_column + XL2_column + XQ_column + XYXZ_column + QYQF_column + SC_column + LMZ_column + FML_column + TJCZ_column + ZBXM_column + FB_column + SPHD_column + PTY_column + XMHJ_column + GWWJ_column + BHKXTT_column + HSLYQ_column + SYZC_column + CDYH_column + XMSL_column + MBFC_column
NODATA_column = XSZJ_column + GCZ_column + LDC_column + FFL_column + YWSL_column + GTSP_column
ALL_column = MAIN_column + NODATA_column
print(len(MAIN_column))
print(len(NODATA_column))
print(len(ALL_column))

# 进入活动，离开活动，进入副本，副本结算，['LOG_COPY_ENDINFO']
EXTRA_column = ['LOG_ACT_START', 'LOG_ACT_END', 'LOG_FB_ENTER', 'LOG_PLAYER_ENTER_MAP', 'LOG_TYPE_TASK_ADDEPT_TASK']
FULL_column = ALL_column + EXTRA_column
print(len(FULL_column))

# 无用数据
NONSENSE_column = ['LOG_TYPE_MALL_SHOP_BUY_START', 'LOG_FB_ENTER', 'LOG_TYPE_COLLECT_GIFT_BEGIN',
                   'LOG_TYPE_COLLECT_GIFT_END', 'LOG_NEW_CATCH_DEVIL_START', 'LOG_NEW_CATCH_DEVIL_AWARD',
                   'LOG_ACT_START', 'LOG_ACT_END', 'LOG_TYPE_TASK_ADDEPT_TASK', 'LOG_MULTI_WAR_BEGIN',
                   'GAME_LOG_XIANMEN_TRAILS_START', 'LOG_PLAYER_ENTER_MAP', 'LOG_ACT_END']

# 大分类用
ALL2_column = ['ZQ_column', 'TGX_column', 'DZ_column', 'FB_column', 'SRZQ_column', 'WG_column', 'SQ_column',
               'YS_column', 'XL_column', 'BH_column', 'JLT_column', 'LJL_column', 'BGY_column', 'XF_column',
               'JL_column', 'SHJN_column', 'FL_column', 'JYH_column', 'JD_column', 'LLZL_column', 'YWSL_column',
               'XL2_column', 'XQ_column', 'SC_column', 'LMZ_column', 'FML_column', 'TJCZ_column', 'ZBXM_column',
               'FB_column', 'SPHD_column', 'PTY_column', 'XMHJ_column', 'GWWJ_column', 'BHKXTT_column', 'HSLYQ_column',
               'SYZC_column', 'CDYH_column', 'XMSL_column', 'MBFC_column']

# 查询有无重复字段
from collections import Counter  # 引入Counter

a = FULL_column
b = dict(Counter(a))
print([key for key, value in b.items() if value > 1])  # 只展示重复元素
print({key: value for key, value in b.items() if value > 1})  # 展现重复元素和重复次数

# 时间参数
BASE_TIME = datetime.datetime.strptime('2020-09-05 00:00:00', '%Y-%m-%d %H:%M:%S')

print('数据开始清洗')
print('loading...')

# 数据预处理
csv_file = pd.read_csv(r'D:\data\data_1.csv')
# print(csv_file)
# print(csv_file.columns)
# print(csv_file.shape)
# print(csv_file.head)

csv_file_clean = csv_file[csv_file['logtype'].isin(FULL_column)]
# print(csv_file_clean.head)

# 数据裁剪
print('开始裁剪')
print('loading...')
csv_file_split = csv_file_clean.iloc[1000000:2000000]
#print(csv_file_split.shape)
#print(csv_file_split.head, csv_file_split.shape)
print('裁剪完毕')

# 数据转换
print('开始数据转换')
print('loading...')
csv_file_split['newlogtime'] = csv_file_split['newlogtime'].map(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") - BASE_TIME).seconds)
#print(csv_file_split.head, csv_file_split.shape)

csv_file_split['logtype'] = csv_file_split['logtype'].map(lambda x: FULL_column.index(x))
#print(csv_file_split.head, csv_file_split.shape)
print('数据转换完毕')
# 每一个用户一个2维数组，600行（时间），N列，
player_column = list(set(list(csv_file_split.iloc[:, 0])))
#print(len(player_column))
print('该数据内包含',end='')
print(len(player_column))
print('人')

#####test
SUM = np.zeros((60 * 60 * 24, 1))
for N in player_column:
    test1 = csv_file_split[csv_file_split['playerid'] == N]
    test1.reset_index(drop=True, inplace=True)
    # print(test1.shape,type(test1))

    # 创建矩阵
    c = np.zeros((60 * 60 * 24, len(FULL_column)))
    # print(c.shape)

    # 商城 LOG_TYPE_MALL_SHOP_BUY_RESULT
    split = FULL_column.index('LOG_TYPE_MALL_SHOP_BUY_RESULT')
    # print(split)

    # 所有片段动作
    test1[test1['logtype'] < split]

    for n in range(len(test1[test1['logtype'] < split])):
        c[test1[test1['logtype'] < split].iloc[n][2]][test1[test1['logtype'] < split].iloc[n][1]] = 1

    # 找出那些logtype为连续动作的数据
    # 连续动作开始
    c_b_s = FULL_column.index('LOG_TYPE_MALL_SHOP_BUY_START')
    # 连续动作结束
    c_b_e = FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_5')
    # print(c_b_s,c_b_e)

    # 所有连续动作
    test2 = test1[(test1['logtype'] >= c_b_s) & (test1['logtype'] <= c_b_e)]

    # t1 end时间 t2 start时间
    for n in range(len(test2)):
        # 判断该数据是玩家商城购买结果
        if test1.iloc[test2.iloc[n].name][1] == FULL_column.index('LOG_TYPE_MALL_SHOP_BUY_RESULT'):
            # t1为玩家商城购买结果的时间
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为玩家开始商城
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_TYPE_MALL_SHOP_BUY_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_TYPE_MALL_SHOP_BUY_RESULT')] = 1
            else:
                pass
        # 判断该数据是玩家进入炼魔阵
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('LOG_TYPE_LMZ'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为进入活动
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_FB_ENTER'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_TYPE_LMZ')] = 1
            # 判断前二个数据为进入活动
            elif int(test1.iloc[test2.iloc[n].name - 2][1]) == FULL_column.index('LOG_FB_ENTER'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_TYPE_LMZ')] = 1
            else:
                pass
        # 判断该数据是玩家进入封魔录
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('LOG_COPY_FML'):
            # t1为玩家商城购买结果的时间
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为进入副本
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_FB_ENTER'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_COPY_FML')] = 1
            else:
                pass
        # 判断该数据是玩家进入天降彩珠
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_23'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始采集掉落宝箱
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_TYPE_COLLECT_GIFT_BEGIN'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_23')] = 1
            # 判断前二个数据为开始采集掉落宝箱
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_TYPE_COLLECT_GIFT_BEGIN'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_23')] = 1
            else:
                pass
        # 判断该数据是玩家进入炼追捕邪魔
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_86'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始新追捕邪魔
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_NEW_CATCH_DEVIL_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_86')] = 1
            # 判断前二个数据为开始新追捕邪魔
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_NEW_CATCH_DEVIL_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_86')] = 1
            else:
                pass
        # 判断该数据是玩家进入副本
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('LOG_COPY_ENDINFO'):
            # t1为玩家商城购买结果的时间
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为进入副本
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_FB_ENTER'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_COPY_ENDINFO')] = 1
            else:
                pass
        # 判断该数据是玩家进入赛跑活动
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('LOG_ACT_RUNNING_AWARD'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始赛跑活动
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_ACT_RUNNING_AWARD')] = 1
            # 判断前二个数据为开始赛跑活动
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('LOG_ACT_RUNNING_AWARD')] = 1
            else:
                pass
        # 判断该数据是玩家进入蟠桃园
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_ACT_START_10'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始新追捕邪魔
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_ACT_START_10')] = 1
            # 判断前二个数据为开始新追捕邪魔
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_ACT_START_10')] = 1
            else:
                pass
        # 判断该数据是玩家进入血魔幻境
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_14'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始血魔幻境
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_14')] = 1
            # 判断前二个数据为开始血魔幻境
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_14')] = 1
            else:
                pass
        # 判断该数据是玩家进入怪物围剿
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_73'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始怪物围剿
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_TYPE_TASK_ADDEPT_TASK'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_73')] = 1
            # 判断前二个数据为开始怪物围剿
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_TYPE_TASK_ADDEPT_TASK'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_73')] = 1
            else:
                pass
        # 判断该数据是玩家进入保护苦行头陀
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_7'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始怪物围剿
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_7')] = 1
            # 判断前二个数据为开始怪物围剿
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_7')] = 1
            else:
                pass
        # 判断该数据是玩家进入护送李英琼
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_4'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始怪物围剿
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_4')] = 1
            # 判断前二个数据为开始怪物围剿
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_4')] = 1
            else:
                pass
        # 判断该数据是玩家进入圣焰战场
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_13'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始怪物围剿
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_MULTI_WAR_BEGIN'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_13')] = 1
            # 判断前二个数据为开始怪物围剿
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_MULTI_WAR_BEGIN'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_13')] = 1
            else:
                pass
        # 判断该数据是玩家进入超度游魂
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_69'):
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为开始怪物围剿
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_69')] = 1
            # 判断前二个数据为开始怪物围剿
            elif test1.iloc[test2.iloc[n].name - 2][1] == FULL_column.index('LOG_ACT_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_69')] = 1
            else:
                pass
        # 判断该数据是玩家进入仙门试炼
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('GAME_LOG_XIANMEN_TRAILS'):
            # t1为仙门试炼开始
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为仙门试炼开始
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('GAME_LOG_XIANMEN_TRAILS_START'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('GAME_LOG_XIANMEN_TRAILS')] = 1
            else:
                pass
        # 判断该数据是玩家进入秘宝蒙尘
        elif test1.iloc[test2.iloc[n].name][1] == FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_5'):
            # t1为进入秘宝蒙尘的时间
            t1 = test1.iloc[test2.iloc[n].name][2]
            # 判断前一个数据为玩家地图切换
            if test1.iloc[test2.iloc[n].name - 1][1] == FULL_column.index('LOG_PLAYER_ENTER_MAP'):
                t2 = test1.iloc[test2.iloc[n].name - 1][2]
                c[t2:t1, FULL_column.index('CHILDLOG_LOG_SKYBOOK_ADD_TIMES_5')] = 1
            else:
                pass
        else:
            pass

    # 转成DATAFRAME
    c = pd.DataFrame(c)

    # 全转为T/F
    c = c > 0
    # print(c)
    # 创建一个新的表
    d = np.zeros((60 * 60 * 24, len(ALL2_column)))
    # print(d.shape)
    # 将几列数值相加

    #     for t in ALL2_column[:2]:
    #         ttt1 = d[:,ALL2_column.index(t)]
    #         print(ttt1.shape )
    #         FULL_column.index(n) for n in t

    d[:, ALL2_column.index('ZQ_column')] = np.sum(c[[FULL_column.index(n) for n in ZQ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('TGX_column')] = np.sum(c[[FULL_column.index(n) for n in TGX_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('DZ_column')] = np.sum(c[[FULL_column.index(n) for n in DZ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('FB_column')] = np.sum(c[[FULL_column.index(n) for n in FB_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('SRZQ_column')] = np.sum(c[[FULL_column.index(n) for n in SRZQ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('WG_column')] = np.sum(c[[FULL_column.index(n) for n in TGX_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('SQ_column')] = np.sum(c[[FULL_column.index(n) for n in SQ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('YS_column')] = np.sum(c[[FULL_column.index(n) for n in YS_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('XL_column')] = np.sum(c[[FULL_column.index(n) for n in XL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('BH_column')] = np.sum(c[[FULL_column.index(n) for n in BH_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('JLT_column')] = np.sum(c[[FULL_column.index(n) for n in JLT_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('LJL_column')] = np.sum(c[[FULL_column.index(n) for n in LJL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('BGY_column')] = np.sum(c[[FULL_column.index(n) for n in BGY_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('XF_column')] = np.sum(c[[FULL_column.index(n) for n in XF_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('JL_column')] = np.sum(c[[FULL_column.index(n) for n in JL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('SHJN_column')] = np.sum(c[[FULL_column.index(n) for n in SHJN_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('FL_column')] = np.sum(c[[FULL_column.index(n) for n in FL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('JYH_column')] = np.sum(c[[FULL_column.index(n) for n in JYH_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('JD_column')] = np.sum(c[[FULL_column.index(n) for n in JD_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('LLZL_column')] = np.sum(c[[FULL_column.index(n) for n in LLZL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('YWSL_column')] = np.sum(c[[FULL_column.index(n) for n in YWSL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('XL2_column')] = np.sum(c[[FULL_column.index(n) for n in XL2_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('XQ_column')] = np.sum(c[[FULL_column.index(n) for n in XQ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('SC_column')] = np.sum(c[[FULL_column.index(n) for n in SC_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('LLZL_column')] = np.sum(c[[FULL_column.index(n) for n in LLZL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('LMZ_column')] = np.sum(c[[FULL_column.index(n) for n in LMZ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('FML_column')] = np.sum(c[[FULL_column.index(n) for n in FML_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('TJCZ_column')] = np.sum(c[[FULL_column.index(n) for n in TJCZ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('ZBXM_column')] = np.sum(c[[FULL_column.index(n) for n in ZBXM_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('FB_column')] = np.sum(c[[FULL_column.index(n) for n in FB_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('SPHD_column')] = np.sum(c[[FULL_column.index(n) for n in SPHD_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('PTY_column')] = np.sum(c[[FULL_column.index(n) for n in PTY_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('XMHJ_column')] = np.sum(c[[FULL_column.index(n) for n in XMHJ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('GWWJ_column')] = np.sum(c[[FULL_column.index(n) for n in GWWJ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('BHKXTT_column')] = np.sum(c[[FULL_column.index(n) for n in BHKXTT_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('HSLYQ_column')] = np.sum(c[[FULL_column.index(n) for n in HSLYQ_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('SYZC_column')] = np.sum(c[[FULL_column.index(n) for n in SYZC_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('CDYH_column')] = np.sum(c[[FULL_column.index(n) for n in CDYH_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('XMSL_column')] = np.sum(c[[FULL_column.index(n) for n in XMSL_column]], axis=1) > 0 + 0
    d[:, ALL2_column.index('MBFC_column')] = np.sum(c[[FULL_column.index(n) for n in MBFC_column]], axis=1) > 0 + 0

    # print(d[:,ALL2_column.index('ZQ_column')])
    # 删除部分特征
    # c = np.delete(c,[],axis=1)

    # print(d.shape,d[:, 0].shape)

    # 转置
    # d = d.transpose()
    # 降维

    # print(c.shape)
    #pca = PCA(n_components=1)
    #pca_d = pca.fit_transform(d)
    # print(type(pca_d),pca_d.shape)

    # print(pca_d)

    SUM = np.hstack((SUM,  d))
    # print(SUM.shape)

# dbscan = DBSCAN(eps = 0.5,min_samples =1000)
# dbscan.fit(pca_d)
# label_pred = dbscan.labels_

# x0 = pca_d[label_pred == 0]
# x1 = pca_d[label_pred == 1]

# plt.scatter(x0[:, 0], x0[:, 1], c = "r", marker='o', label='label0')
# plt.scatter(x1[:, 0], x1[:, 1], c = "g", marker='*', label='label1')

# plt.show()

# 删除第一列全0
SUM = np.delete(SUM, 0, axis=1)
print(SUM.shape)

print('数据清洗完毕')

# kmeans
SSE = []  # 存放每次结果的误差平方和
for k in range(1, 20):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(SUM)
    SSE.append(estimator.inertia_)

X = range(1, 20)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE)

start = datetime.datetime.now()

print(SUM.shape)
pca = PCA(n_components=3)
pca_SUM = pca.fit_transform(SUM)
print(pca_SUM, pca_SUM.shape)
# for n in range(10):
kmodel = KMeans(n_clusters=2)
kmodel.fit(SUM)

cluster_labels = kmodel.labels_

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

LABEL_COLOR_MAP = {0: 'r',
                   1: 'k',
                   2: 'g',
                   3: 'y',
                   4: 'pink',
                   5: 'b'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in cluster_labels]

xs = pca_SUM[:, 0]
ys = pca_SUM[:, 1]
zs = pca_SUM[:, 2]
ax.scatter(xs, ys, zs, c=label_color)

ax.set_xlabel('pca_1')
ax.set_ylabel('pca_2')
ax.set_zlabel('pca_3')

plt.show()

# print(metrics.calinski_harabasz_score(pca_X, y_pred))
end = datetime.datetime.now()
print(end - start)
