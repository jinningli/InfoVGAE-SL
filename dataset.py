import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import json
import pandas
import pickle
from feature_builder import MFFeatureBuilder, TfidfEmbeddingVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=5)

class DatasetBase():
    def __init__(self):
        self.name = "DatasetBase"

    # tokenize filter
    def lenFilter(self, word):
        return len(word) >= 2

    def tokenize(self, text, stopwords=[], keyword=[]):
        # get rid of URL
        original_text = str(text).lower()
        tok = original_text.split(' ')
        text = u''
        if len(tok) == 1:
            for x in tok:
                text = text + ' ' + x
        else:
            for x in tok:
                if len(keyword) > 0:
                    if x not in keyword: continue
                elif len(stopwords) > 0:
                    if len(x) == 0:
                        continue
                    elif (x[0:4] == 'http' or x[0:5] == 'https') and len(tok) != 1:
                        continue
                    elif x[0] == '@':
                        continue
                    elif x in stopwords:
                        continue
                text = text + ' ' + x
        translate_to = u' '

        word_sep = u" ,.?:;'\"/<>`!$%^&*()-=+~[]\\|{}()\n\t" \
                   + u"©℗®℠™،、⟨⟩‒–—―…„“”–――»«›‹‘’：（）！？=【】 ・" \
                   + u"⁄·† ‡°″¡¿÷№ºª‰¶′″‴§|‖¦⁂❧☞‽⸮◊※⁀「」﹁﹂『』﹃﹄《》―—" \
                   + u"“”‘’、，一。►…¿«「」ー⋘▕▕▔▏┈⋙一ー।;!؟"
        word_sep = u'#' + word_sep
        translate_table = dict((ord(char), translate_to) for char in word_sep)
        tokens = text.translate(translate_table).split(' ')
        return ' '.join(sorted(list(filter(self.lenFilter, tokens))))

    # from rawTweet to clean keyword text
    def textProcess(self, data, keyword_path, stopword_path, kthreshold, uthreshold, tthreshold):
        # print(len(data), kthreshold, uthreshold, tthreshold)
        # print(len(data), len(data.name.unique()))
        stopwords = []
        keyword = []
        if keyword_path == 'N':
            # get stopwords
            with open(stopword_path, 'r') as infile:
                for word in infile.readlines():
                    stopwords.append(word[:-1])
            print(len(stopwords))
            assert len(stopwords) != 0

            data['postTweet'] = data.rawTweet.parallel_apply(
                lambda x: self.tokenize(x, stopwords=stopwords, keyword=[]))
            # print(len(data['postTweet'].unique()))
        else:
            # get stopwords
            with open('processed/keyword.txt', 'r') as infile:
                for word in infile.readlines():
                    keyword.append(word[:-1])
            data['postTweet'] = data.rawTweet.parallel_apply(
                lambda x: self.tokenize(x, stopwords=[], keyword=keyword))

        # number of keywords >= 5
        data['keyN'] = data.postTweet.apply(lambda x: len(x.split()))
        data = data[data.keyN >= kthreshold]

        # filter by tweet
        tweet_counts = data['postTweet'].value_counts()
        user_counts = data['name'].value_counts()

        data = data[
            (data['postTweet'].isin(tweet_counts[tweet_counts >= tthreshold].index))  # > into >=
            & (data['name'].isin(user_counts[user_counts >= uthreshold].index))
            ]

        print("After filtering: #Lines: {} #Users: {} #Tweets: {}".format(len(data),
                                                                          len(data["name"].unique()),
                                                                          len(data["postTweet"].unique())))
        data.reset_index(drop=True, inplace=True)

        return data

    def getAdjMatrix(self, nameList, user2id, csv_path, add_self_loop=True, directed=True):
        friend = pd.read_csv(csv_path, sep='\t')

        adjTable = sp.lil_matrix((len(nameList), len(nameList)))

        for u1, u2 in friend.values:
            if (u1 in nameList) and (u2 in nameList):
                adjTable[user2id[u1], user2id[u2]] += 1
                if not directed:
                    adjTable[user2id[u2], user2id[u1]] += 1
        if add_self_loop:
            adjTable += sp.diags([1.0], shape=(len(nameList), len(nameList)))
        # A = A / A.sum(axis=1).reshape(-1, 1)
        # A = (A + np.diag(np.ones(A.shape[0]))) / 2
        return adjTable

    def build(self):
        raise NotImplementedError(self.name)

    def get_feature_similarity_matrix(self):
        raise NotImplementedError(self.name)


class MFDataset(DatasetBase):
    def __init__(self, csv_path, friend_path, keyword_path, stopword_path,
                 mode="multiply", kthreshold=1, uthreshold=-1, tthreshold=-1, num_process=40,
                 add_self_loop=True, directed=True, args=None):
        super(MFDataset).__init__()
        self.name = "MFDataset"
        self.csv_path = csv_path
        self.kthreshold = kthreshold
        self.uthreshold = uthreshold
        self.tthreshold = tthreshold
        self.mode = mode
        self.keyword_path = keyword_path
        self.stopword_path = stopword_path
        self.num_process = num_process
        self.friend_path = friend_path
        self.add_self_loop = add_self_loop
        self.directed = directed
        self.args = args

        self.feature_builder = None

    def build(self):
        print("{} Building...".format(self.name))
        data = pd.read_csv(self.csv_path, sep='\t')
        processed_data = self.textProcess(data, self.keyword_path, self.stopword_path, self.kthreshold, self.uthreshold)
        # Dump tweet label for eval
        print("dump_assertion_label")
        self.dump_assertion_label(processed_data)
        print("dump_assertion_label done")
        self.feature_builder = MFFeatureBuilder(processed_data=processed_data, mode=self.mode,
                                                num_process=self.num_process)
        features = self.feature_builder.build()
        print("feature_builder done")
        user2index = self.feature_builder.user2index
        name_list = processed_data.name.unique().tolist()
        print("processed_data.name.unique().tolist() done")
        adj_matrix = self.getAdjMatrix(name_list, user2index, self.friend_path,
                                       add_self_loop=self.add_self_loop, directed=self.directed)
        print("{} Processing Done".format(self.name))
        return adj_matrix, features, None, name_list

    def dump_assertion_label(self, processed_data):
        labels = []
        with open("dataset/eurovision/annotations_asser_label.pkl", "rb") as fin:
            tweet2label = pickle.load(fin)
        failed = 0
        for i, assertion in enumerate(processed_data.postTweet.unique()):
            asser = assertion.replace(" rt ", " ").replace(" ht ", " ").replace(" in ", " ")
            if asser not in tweet2label.keys():
                failed += 1
                labels.append(4096)
            else:
                labels.append(tweet2label[asser])
        with open(self.args.output_path + "/MF_feature_dim1_label.pkl", "wb") as fout:
            pickle.dump(np.array(labels).astype("int32"), fout)
            print("Assertion Label Dump Success.")

    def get_feature_similarity_matrix(self):
        normed_t2k = self.feature_builder.tweets2keywords / self.feature_builder.tweets2keywords.sum(axis=1).reshape(-1,
                                                                                                                     1)
        # normed_t2k = self.feature_builder.tweets2keywords
        return normed_t2k @ normed_t2k.transpose()


class TwitterDataset(MFDataset):
    def __init__(self, csv_path, keyword_path, stopword_path,
                 mode="multiply", kthreshold=-1, uthreshold=-1, tthreshold=-1, num_process=40,
                 add_self_loop=True, directed=True, args=None):
        super(TwitterDataset, self).__init__(csv_path, None, keyword_path, stopword_path,
                                             mode, kthreshold, uthreshold, tthreshold, num_process, add_self_loop, directed, args)
        self.name = "TwitterDataset"
        self.processed_data = None
        self.user_label = None
        self.asser_label = None
        self.asserlist = None
        self.name_list = None
        self.user_label_numerical = None

        pkl_path = "/".join(args.data_path.split("/")[:-1])
        if not os.path.exists(pkl_path + "/" + "{}TwitterDataset.pkl".format(os.path.basename(args.data_path))):
            print("Preprocess and dump dataset...")
            self.preprocessing()
            with open(pkl_path + "/" + "{}TwitterDataset.pkl".format(os.path.basename(args.data_path)), "wb") as fout:
                pickle.dump([self.data, self.processed_data, self.name_list, self.feature_builder,
                             self.num_user, self.num_assertion, self.num_nodes], fout)
        else:
            print("Use existing dataset pkl {} (Remove file to re-build) ...".format(
                pkl_path + "/" + "{}TwitterDataset.pkl".format(os.path.basename(args.data_path))))
            with open(pkl_path + "/" + "{}TwitterDataset.pkl".format(os.path.basename(args.data_path)), "rb") as fin:
                self.data, self.processed_data, self.name_list, self.feature_builder, \
                self.num_user, self.num_assertion, self.num_nodes = pickle.load(fin)

    def preprocessing(self):
        # Preprocessing
        self.data = pd.read_csv(self.csv_path, sep='\t')
        print("textProcess start")
        self.processed_data = self.textProcess(self.data, self.keyword_path, self.stopword_path, self.kthreshold,
                                               self.uthreshold, self.tthreshold)
        print("textProcess done")
        self.name_list = self.processed_data.name.unique().tolist()
        self.num_user = len(self.processed_data["name"].unique())
        self.num_assertion = len(self.processed_data["postTweet"].unique())
        self.num_nodes = self.num_user + self.num_assertion

        # Feature builder for index mapping
        print("MFFeatureBuilder start")
        self.feature_builder = MFFeatureBuilder(processed_data=self.processed_data, mode=self.mode,
                                                num_process=self.num_process)
        print("MFFeatureBuilder done")
        self.feature_builder.build_index_mapping_only()
        print("build_index_mapping_only done")

    def build(self):
        print("{} Building...".format(self.name))

        # Heterogeneous adjacent matrix
        self.het_matrix = sp.lil_matrix((self.num_nodes, self.num_nodes))

        # Get tweeting matrix
        self.tweeting_matrix = self.get_tweeting_matrix(self.processed_data, self.num_user, self.num_assertion)
        self.het_matrix[:self.num_user, self.num_user:self.num_user + self.num_assertion] = self.tweeting_matrix
        self.het_matrix[self.num_user:self.num_user + self.num_assertion,
        :self.num_user] = self.tweeting_matrix.transpose()
        if self.args.smooth_matrix:
            self.het_matrix[:self.num_user, :self.num_user] = self.compute_u_matrix_same_sparsity(self.tweeting_matrix)
            self.het_matrix[
            self.num_user:self.num_user + self.num_assertion, self.num_user:self.num_user + self.num_assertion
            ] = self.compute_t_matrix_same_sparsity(self.tweeting_matrix)

        # Get following matrix
        if self.args.use_follow:
            following_matrix = self.get_following_matrix()
            self.het_matrix[:self.num_user, :self.num_user] = following_matrix

        print("{} Processing Done".format(self.name))
        # Return adj matrix
        return self.het_matrix

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-09)

    def compute_u_matrix_same_sparsity(self, ut: np.array, method="intersection_union"):
        # ut here should only contains ones or zeros
        if method == "intersection_union":
            intersections = ut @ ut.transpose()
            inverse_ut = 1 - ut
            unions = ut @ inverse_ut.transpose() + ut @ ut.transpose() + inverse_ut @ ut.transpose()
            user_similarity_matrix = intersections / (unions + 1e-09)
        elif method == "cosine_similarity":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        expected_percentile = (ut.shape[0] * ut.shape[1] - np.sum(ut)) / (ut.shape[0] * ut.shape[1]) * 100.0
        expected_threshold = np.percentile(user_similarity_matrix, expected_percentile)
        user_similarity_matrix[user_similarity_matrix < expected_threshold] = 0
        user_similarity_matrix[user_similarity_matrix >= expected_threshold] = 1
        print(user_similarity_matrix)
        print(user_similarity_matrix.shape)
        print(np.sum(user_similarity_matrix))
        print("ut shape: {} ut ones: {}".format(ut.shape, np.sum(ut)))
        return user_similarity_matrix

    def compute_t_matrix_same_sparsity(self, ut: np.array, method="intersection_union"):
        # ut here should only contains ones or zeros, [num_user, num_assertion]
        return self.compute_u_matrix_same_sparsity(ut.transpose(), method=method)

    # Not used
    def get_following_matrix(self):
        assert self.args.use_follow
        assert self.args.follow_path is not None
        following_matrix = np.zeros([self.num_user, self.num_user])
        str_name_list = [str(k) for k in self.name_list]
        with open(self.args.follow_path, "r") as fin:
            for line in fin:
                splts = line.strip().split("	")
                if splts[0] in str_name_list and splts[1] in str_name_list:
                    fr = str_name_list.index(splts[0])
                    to = str_name_list.index(splts[1])
                else:
                    continue
                following_matrix[fr][to] = 1
                following_matrix[to][fr] = 1
        return following_matrix

    # Not used
    def get_tweet_similarity_matrix(self, processed_data):
        X = []
        for tweet in processed_data.postTweet.unique():
            X.append(tweet.split(" "))
        builder = TfidfEmbeddingVectorizer(X=X)
        X_emb = builder.transform(X)
        X_emb = np.array(X_emb)
        similarity_matrix = np.array(cosine_similarity(X_emb, dense_output=True))
        return similarity_matrix

    # Not used
    def get_mention_matrix(self, processed_data, num_user):
        assert self.name_list is not None
        mention_matrix = np.zeros((num_user, num_user))
        for i, item in processed_data.iterrows():
            user_name = item["name"]
            init_tweet = item["rawTweet"]
            splits = init_tweet.split(" ")
            mention_name = None
            if splits[0] == "RT":
                mention_name = splits[1].strip().replace("@", "").replace(":", "")
            if mention_name is not None and user_name in self.name_list and mention_name in self.name_list:
                mention_matrix[self.name_list.index(user_name)][self.name_list.index(mention_name)] += 1
                mention_matrix[self.name_list.index(mention_name)][self.name_list.index(user_name)] += 1
        return mention_matrix

    def get_tweeting_matrix(self, processed_data, num_user, num_assertion):
        tweeting_matrix = np.zeros((num_user, num_assertion))
        for i, item in processed_data.iterrows():
            postTweet = item["postTweet"]
            tweet_index = self.feature_builder.tweet2index[postTweet]
            user_name = item["name"]
            user_index = self.feature_builder.user2index[user_name]
            tweeting_matrix[user_index][tweet_index] += 1
        return tweeting_matrix

    def dump_label(self):
        assert self.processed_data is not None
        # Assume there is a label field in data.csv. 0 for unknown, 1 for trump, 2 for biden
        num_assertion = self.num_assertion

        # calculate tweet assertion label
        self.asser_label = np.zeros(num_assertion).astype("int32")
        self.asserlist = [None for _ in range(num_assertion)]
        for i, item in self.processed_data.iterrows():
            label = item["label"]
            postTweet = item["postTweet"]
            tweet_id = self.feature_builder.tweet2index[postTweet]
            if self.asserlist[tweet_id] is None:
                self.asserlist[tweet_id] = item["rawTweet"]
            if self.asser_label[tweet_id] == 0:
                self.asser_label[tweet_id] = label

        # calculate tweet label
        num_user = self.num_user
        self.user_label = np.zeros(num_user).astype("int32")
        user_label_candidate = [[] for _ in range(num_user)]
        for i, item in self.processed_data.iterrows():
            label = item["label"]
            user_name = item["name"]
            user_index = self.feature_builder.user2index[user_name]
            if label != 0:
                user_label_candidate[user_index].append(label)
        for i in range(num_user):
            if not user_label_candidate[i]:
                self.user_label[i] = 0
            else:
                self.user_label[i] = Counter(user_label_candidate[i]).most_common(1)[0][0]

        print("process numerical...")

        # If do not have numerical label field:
        self.user_label_numerical = np.zeros(num_user)
        if "numerical_label" not in self.processed_data:
            for i in range(num_user):
                if not user_label_candidate[i]:
                    self.user_label_numerical[i] = -214738467
                else:
                    label = Counter(user_label_candidate[i]).most_common(1)[0][0]
                    arr = np.array(user_label_candidate[i]).reshape(-1)
                    purity = np.sum(arr == label) / arr.shape[0]
                    if label == 1:
                        self.user_label_numerical[i] = purity
                    else:
                        self.user_label_numerical[i] = -purity
        else:
            user_label_candidate_numerical = [[] for _ in range(num_user)]
            for i, item in self.processed_data.iterrows():
                label = item["numerical_label"]
                user_name = item["name"]
                user_index = self.feature_builder.user2index[user_name]
                if not pd.isna(label):
                    user_label_candidate_numerical[user_index].append(label)
            for i in range(num_user):
                if not user_label_candidate_numerical[i]:
                    self.user_label_numerical[i] = -214738467
                else:
                    arr = np.array(user_label_candidate_numerical[i]).reshape(-1)
                    self.user_label_numerical[i] = np.average(arr)

        print("process numerical done")

        print(self.user_label_numerical.tolist()[:50])

        # dump label to label.bin
        save_path = self.args.output_path + "/label.bin"
        with open(save_path, "wb") as fout:
            pickle.dump({"user_label": self.user_label, "assertion_label": self.asser_label}, fout)

        # dump the representative assertion list
        save_path = self.args.output_path + "/asserlist.json"
        with open(save_path, "w", encoding="utf-8") as fout:
            json.dump(self.asserlist, fout, indent=2)

        # dump namelist for evaluation
        with open(self.args.output_path + "/namelist.json", 'w') as fout:
            json.dump(self.name_list, fout, indent=2)

        # print("process pickle get_tweeting_matrix start")
        # # dump tweeting matrix for baseline evaluation
        # with open(self.args.output_path + "/tweeting_matrix.bin", 'wb') as fout:
        #     pickle.dump(self.get_tweeting_matrix(self.processed_data, self.num_user, self.num_assertion), fout)
        #
        # print("process pickle get_tweeting_matrix done")

        # dump tweet to assertion id mapping
        tweet2asserid = {}
        for i, item in self.processed_data.iterrows():
            postTweet = item["postTweet"]
            if "id" in item.keys():
                tweet_id = item["id"]
            else:
                tweet_id = item["tweet_id"]
            assertion_id = self.feature_builder.tweet2index[postTweet]
            tweet2asserid[tweet_id] = assertion_id
        with open(self.args.output_path + "/tweet_to_assertion_id_map.json", 'w') as fout:
            json.dump(tweet2asserid, fout, indent=2)

        print("Dump Label file success {}".format(save_path))

    def dump_processed_data(self):
        self.processed_data.to_csv(self.args.output_path + "/" + "processed_data.csv",
                                   sep="\t", encoding="utf-8", index=False)
        print("Dump processed data success {}".format(self.args.output_path + "/" + "processed_data.csv"))


"""
TODO
gaussian mixture
"""


class BillDataset(DatasetBase):
    def __init__(self, args):
        super(BillDataset).__init__()
        self.args = args
        with open("dataset/bill/data_80_115.pkl", "rb") as fin:
            data = pickle.load(fin)
        with open("dataset/bill/memandparty.pkl", "rb") as fin:
            self.party_names, self.icpsr2party = pickle.load(fin)
        with open("dataset/bill/bmap2.pkl", "rb") as fin:
            self.bmap = pickle.load(fin)
        self.data = data[105]  # Use 105th Congress
        self.idx2icpsr, self.idx2bill, self.userbillmat, self.frequsermat = self.data
        self.num_user = self.userbillmat.shape[0]
        self.num_bill = self.num_assertion = self.userbillmat.shape[1]
        self.num_nodes = self.num_user + self.num_bill
        print("User: {}, Bill: {}".format(self.num_user, self.num_bill))
        self.userbillmat_csr = sp.lil_matrix(self.userbillmat)
        self.adj_matrix = sp.lil_matrix((self.num_user + self.num_bill, self.num_user + self.num_bill))
        self.adj_matrix[:self.num_user, self.num_user:self.num_user + self.num_bill] = self.userbillmat_csr
        self.adj_matrix[self.num_user:self.num_user + self.num_bill, :self.num_user] = self.userbillmat_csr.transpose()
        self.features = sp.diags([1.0], shape=(self.num_user + self.num_bill, self.num_user + self.num_bill))

    def build(self):
        return self.adj_matrix

    def dump_label(self):
        # Get user label
        self.user_label = np.zeros(self.num_user).astype("int32")
        self.user_label_numerical = np.zeros(self.num_user)
        ideology_value_pd = pandas.read_csv("./dataset/bill/H105_members.csv")
        ideology_value_mapping = {}
        for i, item in ideology_value_pd.iterrows():
            icpsr = item["icpsr"]
            ideology_value_gt = item["nominate_dim1"]
            ideology_value_mapping[icpsr] = ideology_value_gt
        for i in range(self.num_user):
            label = self.icpsr2party[self.idx2icpsr[i]]
            label_numerical = ideology_value_mapping[self.idx2icpsr[i]]
            if label == 100:
                self.user_label[i] = 1
            elif label == 200:
                self.user_label[i] = 2
                # labels.append(-1)
            else:
                # print(label)
                self.user_label[i] = 0
                # labels.append(0)
            self.user_label_numerical[i] = label_numerical

        # Get bill label
        self.passed = [Counter([x[1] for x in self.bmap[self.idx2bill[i]]]).most_common()[0][0] for i in
                       range(self.num_bill)]
        self.bill_label = np.zeros(self.num_bill).astype("int32")
        for i in range(self.num_bill):
            filt = list(filter(lambda y: y[1] == self.passed[i], self.bmap[self.idx2bill[i]]))
            counter = Counter(map(lambda x: self.icpsr2party[x[0]], filt))
            if self.passed[i] == 1:
                if counter[100] > counter[200]:
                    self.bill_label[i] = 1
                else:
                    self.bill_label[i] = 2
            elif self.passed[i] == 0:
                if counter[100] > counter[200]:
                    self.bill_label[i] = 2
                else:
                    self.bill_label[i] = 1
        self.asser_label = self.bill_label

        # dump label to label.bin
        save_path = self.args.output_path + "/label.bin"
        with open(save_path, "wb") as fout:
            pickle.dump({"user_label": self.user_label, "assertion_label": self.bill_label}, fout)

        # dump the representative assertion list
        self.asserlist = ["Bill {}".format(k) for k in range(self.num_bill)]
        save_path = self.args.output_path + "/asserlist.json"
        with open(save_path, "w", encoding="utf-8") as fout:
            json.dump(self.asserlist, fout, indent=2)

        # dump namelist for evaluation
        self.name_list = ["User {}".format(k) for k in range(self.num_user)]
        with open(self.args.output_path + "/namelist.json", 'w') as fout:
            json.dump(self.name_list, fout, indent=2)
