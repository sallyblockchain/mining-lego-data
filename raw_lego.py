import json
import logging
import random
import math

from collections import defaultdict
from collections import Counter
from itertools import combinations
from operator import itemgetter

# from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
# from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim
import numpy as np
from matplotlib.pyplot import plot, show
from gensim.matutils import corpus2dense
from sklearn.cluster import KMeans


####################
# Logging settings
    
LOG_LEVEL = logging.INFO

logging.basicConfig( level=LOG_LEVEL, format='%(levelname)s %(message)s')
gLogger = logging.getLogger('legosets')

####################

class Lego:

    def __init__(self):
        self.colors = set()
        self.piece_ids = set()
        self.set_ids = set()
        self.meaningful_themes = defaultdict(int)
        # self.all_sets = []
        self.all_sets = {}
        self.set_piece_dict = defaultdict() # for parts
        self.piece_dict = defaultdict(int)
        self.piece_dict_with_color = defaultdict(int)
        self.piece_category = {}
        self.theme_category = defaultdict(int)
        self.training_theme_category = defaultdict(int)
        self.total_piece_vocabulary = 0 
        self.total_piece = 0
        self.training = {}
        self.testing = {}
        self.training_piece_tf = {}
        self.training_piece_idf = {}
        self.training_theme_parts = {}
        self.training_theme_parts_with_color = {}

        self.read_colors("colors.csv")
        self.read_meaningful_themes("meaningful_themes_2004_2014.txt")
        self.read_piece_ids("pieces.tsv")
        self.read_set_pieces("set_pieces.csv")
        self.get_all_sets_json("sets.tsv", "all_sets.json")
        #self.read_training_testing_set("training.json", "testing.json")
        #self.generate_training_theme_parts(True)

    def read_colors(self, INPUT_FILE_NAME):
        gLogger.info( "Reading colors.csv file..." )
        with open(INPUT_FILE_NAME, 'r') as f:
            f.readline()
            for line in f.readlines():
                line_contents = line.split(',')
                id = line_contents[0].strip()
                descr = line_contents[1].strip()
                self.colors.add(id)
        
    def read_piece_ids(self, INPUT_FILE_NAME):
        gLogger.info( "Reading pieces.tsv file..." )
        with open(INPUT_FILE_NAME, 'r') as f:
            f.readline()
            for line in f.readlines():
                line_contents = line.split('\t')
                piece_id = line_contents[0].strip()
                category = line_contents[2].strip()
                self.piece_ids.add(piece_id)
                self.piece_category[piece_id] = category

    def read_meaningful_themes(self, INPUT_FILE_NAME):
        gLogger.info( "Reading meaningful_themes_2004_2014.txt file..." )
        with open(INPUT_FILE_NAME, 'r') as f:
            for line in f.readlines():
                line_contents = line.replace('"', "").strip().split(',')
                theme_name = line_contents[0]
                num = line_contents[1]
                self.meaningful_themes[theme_name] = num
            print "Num of Meaningful Themes: " + str(len(self.meaningful_themes))

    def read_set_pieces(self, INPUT_FILE_NAME):
        gLogger.info( "Reading set_pieces.csv file..." )
        colors = set() 
        types = set()
        with open(INPUT_FILE_NAME, 'r') as f:
            f.readline()
            for line in f.readlines():
                line_contents = line.split(',')
                set_id = line_contents[0].strip()
                self.set_ids.add(set_id)
                piece_id = line_contents[1].strip()
                num = int(line_contents[2].strip())
                color  = line_contents[3].strip()
                # key = (piece_id, color)
                key = piece_id + "+" + color

                if set_id not in self.set_piece_dict:
                    self.set_piece_dict[set_id] = defaultdict()
                    self.set_piece_dict[set_id][key] = num
                else:
                    if key not in self.set_piece_dict[set_id]:
                        self.set_piece_dict[set_id][key] = num
                    else:
                        self.set_piece_dict[set_id][key] += num

                if piece_id not in self.set_piece_dict[set_id]:
                    self.set_piece_dict[set_id][piece_id] = num
                else:
                    self.set_piece_dict[set_id][piece_id] += num
                # if piece_id not in self.piece_ids:
                #     print piece_id

                type = line_contents[4].strip()
                colors.add(color)
                types.add(type)
            # print len(set_ids)
            # print self.set_piece_dict['10220-1']
            # c = 0
            # for each in self.set_piece_dict['10220-1']:
            #     if isinstance(each, str):
            #         c += self.set_piece_dict['10220-1'][each]
            # print c

            # print len(self.piece_dict_with_color)
            # print len(self.piece_dict)
            # print self.piece_dict

    def get_all_sets_json(self, INPUT_FILE_NAME, OUTPUT_FILE_NAME):
        gLogger.info( "Reading sets.tsv file..." )

        with open(INPUT_FILE_NAME, 'r') as in_file:
            in_file.readline()
            c = 0
            # t1 = set()
            for line in in_file.readlines():
                line_contents = line.split('\t')
                set_id = line_contents[0].strip()
                year = int(line_contents[1].strip())
                pieces = int(line_contents[2].strip())
                theme = line_contents[3].strip()
                descr = line_contents[4].strip()
                if year >= 2004 and year <= 2014 and theme in self.meaningful_themes and int(self.meaningful_themes[theme]) >= 10 and pieces >= 10:
                    if set_id in self.set_ids:
                        sets_dict = {'set_id': set_id, 'year': year, 'num_pieces': pieces, 'theme': theme, 'descrption': descr}
                        parts_dict = self.set_piece_dict[set_id]
                    
                        sets_dict['parts'] = { key: value for key, value in parts_dict.items() if '+' not in key }
                        sets_dict['parts_with_color'] = { key: value for key, value in parts_dict.items() if '+' in key }
                        sets_dict['num_pieces_plus_spare'] = sum(sets_dict['parts'].values())
                        # self.all_sets.append(sets_dict)
                        self.all_sets[set_id] = sets_dict
                        self.theme_category[theme] += 1
                        # t1.add(theme)
                        c += 1
                    # else:
                    #     print set_id
            # print len(t1)
            print self.theme_category
            # for each in self.all_sets:
            #     out_file.write(json.dumps(each) + "\n")

            
            # with open(OUTPUT_FILE_NAME, 'w') as out_file:
            #     gLogger.info( "Writing all_sets.json file..." )
            #     for each in self.all_sets:
            #          out_file.write(json.dumps(self.all_sets[each]) + "\n")
            
            # print self.all_sets
            print len(self.all_sets)

    def get_piece_categories(self):
        """
        category = defaultdict(int)
        ids = [x for x in self.all_sets]
        result = self.merge_parts_dicts(ids, False)
        for p in result:
            category[self.piece_category[p]] += result[p]
        print category
        """
        documents = []
        doc_for_name_check = []
        i = 0
        for each_set in self.all_sets:
            category = defaultdict(int)
            document = []
            descr = self.all_sets[each_set]['descrption']
            parts_dict = self.all_sets[each_set]['parts']
            for p in parts_dict:
                c = self.piece_category[p]
                category[c] += parts_dict[p]
            for each_c in category:
                document += [each_c] * category[each_c]
            # if each_set == "30201-1":
            #     print document
            #     print len(document)
            documents.append(document)
            doc_for_name_check.append((i, each_set, descr))
            i += 1
        print doc_for_name_check[1984]
        print doc_for_name_check[636]
        # print len(documents)
        dictionary = corpora.Dictionary(documents)
        # print dictionary
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        # print len(corpus)
        # print corpus[0]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
        topics = ldamodel.print_topics(num_topics=5)
        print [t[1] for t in topics]
        print ldamodel.get_document_topics(dictionary.doc2bow(documents[1984]))
        print ldamodel.get_document_topics(dictionary.doc2bow(documents[636]))

    '''
    If it's greater than the threshold, remove the brick.
    '''
    def helper_stop_bricks(self, threshold):
        p_dict = defaultdict(int)
        for each_set in self.all_sets:
            parts_dict = self.all_sets[each_set]['parts']
            for p in parts_dict:
                p_dict[p] += 1
        # sorted_list = sorted(p_dict.items(), key=itemgetter(1), reverse=True)
        # print len([x[0] for x in sorted_list if x[1] < len(self.all_sets) * threshold])
        result = [x for x in p_dict.items() if x[1] >= len(self.all_sets) * threshold]
        return dict(result)

    '''
    If it's greater than the threshold, remove the brick.
    '''
    def get_piece_categories_remove_stop_bricks(self, threshold, topic_k):
        remove_bricks = self.helper_stop_bricks(threshold)
        documents = []
        doc_for_name_check = []
        i = 0
        themes_dict = {}
        for each_set in self.all_sets:
            category = defaultdict(int)
            document = []
            descr = self.all_sets[each_set]['descrption']
            theme = self.all_sets[each_set]['theme']
            if theme not in themes_dict:
                themes_dict[theme] = []
            themes_dict[theme].append(i)
            parts_dict = self.all_sets[each_set]['parts']
            for p in parts_dict:
                if p not in remove_bricks:
                    c = self.piece_category[p]
                    category[c] += parts_dict[p]
            for each_c in category:
                document += [each_c] * category[each_c]
            # if each_set == "30201-1":
            #     print category
            #     print document
            #     print len(document)
            documents.append(document)
            doc_for_name_check.append((i, each_set, theme, descr))
            i += 1


        print doc_for_name_check[1984]
        print doc_for_name_check[636]
        # print len(documents)
        dictionary = corpora.Dictionary(documents)
        # print dictionary
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        # print len(corpus)
        # print corpus[0]

        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_k, id2word = dictionary, passes=20)
        topics = ldamodel.print_topics(num_topics=topic_k)
        print [t[1] for t in topics]
        print 
        print "--------------------------------------------------------"
        id = themes_dict['Monster Fighters'][0]
        for t in themes_dict:
            sets = themes_dict[t]
            distribution = []
            for index in sets:
                d = ldamodel.get_document_topics(dictionary.doc2bow(documents[index]))
                distribution.append(d)
            sum_dist = sum((Counter(dict(x)) for x in distribution), Counter())
            for x in sum_dist:
                sum_dist[x] /= float(len(sets))
            print t, sorted(dict(sum_dist).items(), key=itemgetter(1), reverse=True) 


        print sorted(ldamodel.get_document_topics(dictionary.doc2bow(documents[1984])), key=itemgetter(1), reverse=True)
        print sorted(ldamodel.get_document_topics(dictionary.doc2bow(documents[636])), key=itemgetter(1), reverse=True)
        print sorted(ldamodel.get_document_topics(dictionary.doc2bow(documents[1399])), key=itemgetter(1), reverse=True) #71006
        print sorted(ldamodel.get_document_topics(dictionary.doc2bow(documents[1470])), key=itemgetter(1), reverse=True) #10220
        print sorted(ldamodel.get_document_topics(dictionary.doc2bow(documents[1974])), key=itemgetter(1), reverse=True)

        print 
        print "--------------------------------------------------------"
        new_dict = dict(dictionary.items())
        # for i in range(6):
        #     sorted_l = sorted(ldamodel.get_topic_terms(i), key=itemgetter(1), reverse=True)
        #     print [(new_dict[x[0]], x[1]) for x in sorted_l]
        corpus_lda = ldamodel[corpus]
        index = similarities.MatrixSimilarity(corpus_lda)
        # print index

        l = []
        for s in index:
            l.append(s)
        new_dense = np.asarray(l) 
        print new_dense
        with open('sim.txt', 'w') as f:
            for i in range(len(new_dense)):
                for j in range(len(new_dense[0])):
                    if i < j and new_dense[i][j] >= 0.98:
                        if doc_for_name_check[i][2] != doc_for_name_check[j][2]:
                            result1 = ", ".join('%s' % x for x in doc_for_name_check[i])
                            result2 = ", ".join('%s' % x for x in doc_for_name_check[j])
                            f.write(result1 + '\t' + result2 + '\t' + str(new_dense[i][j]) + '\n')

        km = KMeans(n_clusters=43, init='k-means++', max_iter=100, n_init=1, verbose=1)
        labels =  km.fit(new_dense).labels_
        cluster_list = list(labels)
        clusters = {}
        for i in range(0, len(cluster_list)):
            cluster_number = cluster_list[i]
            if cluster_number not in clusters:
                clusters[cluster_number] = []
            clusters[cluster_number].append(doc_for_name_check[i])
        print cluster_list[1984]
        print cluster_list[636]
        # print [len(clusters[x]) for x in clusters]
        
        for i in clusters:
            sets = clusters[i]
            c = Counter([x[2] for x in sets])
            print i, max(c, key=c.get)
       
        themes = {}
        for i in clusters:
            sets = clusters[i]
            for s in sets:
                theme = s[2]
                if theme not in themes:
                    themes[theme] = defaultdict(int)
                themes[theme][i] += 1
        # print themes
        similar_themes = {}
        for theme in themes:
            c = themes[theme]
            max_value = max(c.values())  #<-- max of values
            print theme,
            max_keys = [key for key in c if c[key] == max_value]
            print max_keys
            for m in max_keys:
                if m not in similar_themes:
                    similar_themes[m] = []
                similar_themes[m].append(theme)
        print similar_themes

        result = {
            "name": "legos",
            "children": []
        }
        with open('data.json', 'w') as f:
            for i in similar_themes:
                r = {
                    "names": str(i),
                    "children": []
                }
                legos = similar_themes[i]
                for l in legos:
                    r["children"].append({"name": l, "size": 1000/len(legos)})
                result["children"].append(r)
            json.dump(result, f)
        
        
    '''
    exact_bool is True when considering the color of the piece;
    exact_bool is False when not considering the color of the piece.
    '''
    def compare_two_sets(self, set_id_1, set_id_2, exact_bool=True):
        # gLogger.info( "Comparing two sets " + set_id_1 + \
        #              " and " + set_id_2 + "..." )
        set_1_info = self.all_sets[set_id_1]
        set_2_info = self.all_sets[set_id_2]
        exact = ""
        if not exact_bool:
            set_1_parts = set_1_info['parts']
            set_2_parts = set_2_info['parts']
        else:
            set_1_parts = set_1_info['parts_with_color']
            set_2_parts = set_2_info['parts_with_color']
            exact = "With Color "
        common_parts_dict = dict([(k, min(v, set_2_parts[k])) for (k, v) in set_1_parts.iteritems() if k in set_2_parts])
        common_parts_num = sum(common_parts_dict.values()) # num of pieces
        common_parts_unique = len(common_parts_dict) # unique

        total_parts_1 = set_1_info['num_pieces_plus_spare']
        total_parts_2 = set_2_info['num_pieces_plus_spare']
        prob_1 = float(common_parts_num) / total_parts_1
        prob_2 = float(common_parts_num) / total_parts_2
        # print "Common parts number: " + str(common_parts_num)
        # print "Common unique parts number: " + str(common_parts_unique)
        # print exact + "Common parts in " + set_id_1 + ": " + str(self.to_percent(prob_1))
        # print exact + "Common parts in " + set_id_2 + ": " + str(self.to_percent(prob_2))
        return prob_1

    def to_percent(self, float_num):
        return "{0:.0f}%".format(float_num * 100)

    '''
    exact_bool is True when considering the color of the piece;
    exact_bool is False when not considering the color of the piece.
    '''
    def compare_with_all(self, set_id_1, exact_bool=True):
        gLogger.info( "Comparing set " + set_id_1 + \
                      " with all other sets..." )
        set_1_info = self.all_sets[set_id_1]
        result = []
        c = 0
        for each in self.all_sets.keys():
            if c % 500 == 0:
                print c
            if each != set_id_1:
                prob = self.compare_two_sets(set_id_1, each, exact_bool)
                result.append((each, prob))
            c += 1
        print sorted(result, key=itemgetter(1), reverse=True)

    '''
    exact_bool is True when considering the color of the piece;
    exact_bool is False when not considering the color of the piece.
    '''
    def merge_parts_dicts(self, set_id_args, exact_bool=True):
        parts_dicts_list = []
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'
        for set_id in set_id_args:
            parts_dicts_list.append(self.all_sets[set_id][keyword])
        result = Counter(parts_dicts_list[0])
        for i in range(1, len(parts_dicts_list)):
            if i % 500 == 0:
                print i
            result += Counter(parts_dicts_list[i])
        result = dict(result)
        return result

    '''
    exact_bool is True when considering the color of the piece;
    exact_bool is False when not considering the color of the piece.
    '''
    def get_piece_freq(self, exact_bool=True, update_file=False):
        set_id_args = self.training.keys()
        result = self.merge_parts_dicts(set_id_args, exact_bool)
        self.total_piece_vocabulary = len(result)
        self.total_piece = sum(result.values())
        print self.total_piece_vocabulary
        print self.total_piece
        
        if update_file:
            if exact_bool:
                OUTPUT_FILE_NAME = 'piece_freq_with_color.json'
            else:
                OUTPUT_FILE_NAME = 'piece_freq.json'
            with open(OUTPUT_FILE_NAME, 'w') as f:
                gLogger.info( "Writing " + OUTPUT_FILE_NAME + " file..." )
                json.dump(result, f)
        return result

    def get_all_parts(self, exact_bool=True):
        all_parts = set()
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'
        for set_id in self.all_sets:
            set_info = self.all_sets[set_id]
            parts = set_info[keyword]
            for p in parts:
                all_parts.add(p)
        return list(all_parts)

    def get_piece_companions(self, exact_bool=True):
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'
        #all_parts = self.get_all_parts(exact_bool)
        #parts_comb = list(combinations(all_parts, 2))
        #print len(parts_comb)
        parts_comb_dict = defaultdict(int)
        c = 0
        for set_id in self.all_sets:
            if c % 500 == 0:
                print c
            set_info = self.all_sets[set_id]
            parts = set_info[keyword].keys()
            pairs = list(combinations(parts, 2))
            for pair in pairs:
                parts_comb_dict[tuple(sorted(pair))] += 1
            c += 1
        # print len(parts_comb_dict)
        sorted_list = sorted(parts_comb_dict.items(), key=lambda x:x[1], reverse=True)
        return sorted_list
 

    '''
    exact_bool is True when considering the color of the piece;
    exact_bool is False when not considering the color of the piece.
    '''
    def get_piece_freq_document(self, exact_bool=True, update_file=False):
        result = {}
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'

        for set_id in self.all_sets:
            theme = self.all_sets[set_id]['theme']
            for p in self.all_sets[set_id][keyword]:
                if p not in result:
                    if '+' in p:
                        new_p = p[:p.find('+')]
                        category = self.piece_category[new_p]
                    else:
                        category = self.piece_category[p]
                    result[p] = { 'piece_id': p, 'total': 1, 'category': category, 'in_themes': {}, 'in_themes_num':0, '10_frequent_companions': [], 'frequent_in_theme': [], 'in_sets_percentage': ""  }
                else:
                    result[p]['total'] += 1
                if theme not in result[p]['in_themes']:
                    result[p]['in_themes'][theme] = 1
                else:
                    result[p]['in_themes'][theme] += 1
                result[p]['in_themes_num'] = len(result[p]['in_themes'])
                max_value = max(result[p]['in_themes'].values())  #<-- max of values
                max_keys = [key for key in result[p]['in_themes'] if result[p]['in_themes'][key] == max_value]
                result[p]['frequent_in_theme'] = max_keys
                result[p]['in_sets_percentage'] = str(round(float(result[p]['total'])*100, 2) / len(self.all_sets)) + "%"

        companion_list = self.get_piece_companions(exact_bool)
        c = 0
        for p in result:
            if result[p]['total'] < 3:
                continue
            if c % 1000 == 0:
                print c
            companions = result[p]['10_frequent_companions']
            for pair_freq in companion_list:
                pair = pair_freq[0]
                freq = pair_freq[1]
                if len(companions) < 10 and p in pair:
                    index = pair.index(p)
                    new_tuple = (pair[1-index], freq)
                    companions.append(new_tuple)
            print companions
            c += 1

        if update_file:
            if exact_bool:
                OUTPUT_FILE_NAME = 'doc_piece_freq_with_color.json'
            else:
                OUTPUT_FILE_NAME = 'doc_piece_freq.json'
            with open(OUTPUT_FILE_NAME, 'w') as f:
                gLogger.info( "Writing " + OUTPUT_FILE_NAME + " file..." )
            #     json.dump(result, f)
                for each in result:
                    f.write(json.dumps(result[each]) + "\n")
        return result

    '''
    Generate once
    '''
    def get_training_testing_set(self, OUTPUT_FILE_NAME_1, 
                                OUTPUT_FILE_NAME_2):
        D = self.all_sets
        keys =  list(D.keys())
        random.shuffle(keys)
        new_D = dict([(key, D[key]) for key in keys])
        theme_copy = dict(self.theme_category)
        theme_training = {x : int(round(theme_copy[x]*0.75)) for x in theme_copy}
        # print theme_training
        for set_id in new_D:
            theme = new_D[set_id]['theme']
            if theme_training[theme] > 0:
                self.training[set_id] = new_D[set_id]
                theme_training[theme] -= 1
            else:
                self.testing[set_id] = new_D[set_id]
        # K = int(round(len(D) * 0.75))
        # training = random.sample( D.items(), K )
        # self.training = dict(training)
        # for set_id in self.all_sets:
        #     if set_id not in self.training:
        #         self.testing[set_id] = self.all_sets[set_id]
        with open(OUTPUT_FILE_NAME_1, 'w') as out_file:
            gLogger.info( "Writing training.json file..." )
            for each in self.training:
                out_file.write(json.dumps(self.training[each]) + "\n")
        with open(OUTPUT_FILE_NAME_2, 'w') as out_file:
            gLogger.info( "Writing testing.json file..." )
            for each in self.testing:
                out_file.write(json.dumps(self.testing[each]) + "\n")

    def read_training_testing_set(self, INPUT_FILE_NAME_1, 
                                INPUT_FILE_NAME_2):
        with open(INPUT_FILE_NAME_1, 'r') as f:
            gLogger.info( "Reading training.json file..." )
            for line in f.readlines():
                line_contents = json.loads(line)
                set_id = line_contents['set_id']
                self.training[set_id] = line_contents
                theme = line_contents['theme']
                self.training_theme_category[theme] += 1

        # print len(self.training) # 1964
        print self.training_theme_category

        with open(INPUT_FILE_NAME_2, 'r') as f:
            gLogger.info( "Reading testing.json file..." )
            for line in f.readlines():
                line_contents = json.loads(line)
                set_id = line_contents['set_id']
                self.testing[set_id] = line_contents
        # print len(self.testing) # 648

    def tf_idf(self, tf, doc_len, N, df):
        # tf_result = 1 + math.log(float(tf) / doc_len)
        tf_result = float(tf) / doc_len
        idf_result = math.log(N / df)
        return tf_result * idf_result

    def get_popular_piece_of_each_set(self, n, exact_bool=True):
        self.training_piece_tf = dict(self.get_piece_freq(exact_bool, False))
        self.training_piece_idf = dict(self.get_piece_freq_document(exact_bool, False))
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'
        for set_id in self.training:
            print "set_id: " + set_id
            set_info = self.training[set_id]
            scores = defaultdict(float)
            for p in set_info[keyword]:
                tf = set_info[keyword][p]
                doc_len = set_info['num_pieces_plus_spare']
                N = len(self.training)
                df = self.training_piece_idf[p]['total']
                scores[p] = self.tf_idf(tf, doc_len, N, df)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for p, score in sorted_scores[:n]:
                print("Piece: {}, TF-IDF: {}".format(p, round(score, 5)))

    '''
    n is number of pieces selected to form a tuple
    '''
    def get_theme_piece(self, n, top, theme_input, exact_bool=True):
        theme_piece = {}
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'
        c = 0
        for set_id in self.training:
            set_info = self.training[set_id]
            theme = set_info['theme']
            if theme not in theme_piece:
                theme_piece[theme] = defaultdict(int)
            keys = set_info[keyword].keys()
            subsets = list(combinations(keys, n))
            for t in subsets:
                theme_piece[theme][tuple(sorted(t))] += 1
            c += 1
            if c % 100 == 0:
                print c
        result = sorted(theme_piece[theme_input].items(), key=lambda x:x[1], reverse=True)[:top]
        print result

    '''
    n is the number of top elements returned
    '''
    def get_popular_piece_of_theme(self, n, theme, exact_bool=True):
        if exact_bool:
            keyword = 'parts_with_color'
        else:
            keyword = 'parts'
        self.training_piece_idf = dict(self.get_piece_freq_document(exact_bool, False))
        
        for set_id in self.training:
            set_info = self.training[set_id]
            t = set_info['theme']
            if t != theme:
                continue
            print "set_id: " + set_id 
            scores = defaultdict(float)
            for p in set_info[keyword]:
                tf = set_info[keyword][p]
                doc_len = set_info['num_pieces_plus_spare']
                N = self.training_theme_category[theme]
                df = self.training_piece_idf[p]['in_themes'][theme]
                scores[p] = self.tf_idf(tf, doc_len, N, df)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for p, score in sorted_scores[:n]:
                print("Piece: {}, TF-IDF: {}".format(p, round(score, 5)))


    def generate_training_theme_parts(self, update_file=False):
        gLogger.info( "Generating training theme parts..." )
        for set_id in self.training:
            set_info = self.training[set_id]
            theme = set_info['theme']
            parts = set_info['parts']
            parts_with_color = set_info['parts_with_color']
            if theme not in self.training_theme_parts:
                self.training_theme_parts[theme] = {'theme_name': theme, 'parts': Counter({})}
            if theme not in self.training_theme_parts_with_color:
                self.training_theme_parts_with_color[theme] = {'theme_name': theme, 'parts_with_color': Counter({})}
            self.training_theme_parts[theme]['parts'] += Counter(parts)
            self.training_theme_parts_with_color[theme]['parts_with_color'] += Counter(parts_with_color)

        print len(self.training_theme_parts)
        print len(self.training_theme_parts_with_color)
        if update_file:
            with open('training_theme_parts.json', 'w') as f:
                gLogger.info( "Writing training_theme_parts.json file..." )
                for each in self.training_theme_parts:
                    f.write(json.dumps(self.training_theme_parts[each]) + "\n")

            with open('training_theme_parts_with_color.json', 'w') as f:
                gLogger.info( "Writing training_theme_parts_with_color.json file..." )
                for each in self.training_theme_parts_with_color:
                    f.write(json.dumps(self.training_theme_parts_with_color[each]) + "\n")

    '''
    n is the number of top elements returned
    '''
    def get_popular_piece_of_theme_new(self, n, theme, exact_bool=True):
        if exact_bool:
            keyword = 'parts_with_color'
            theme_dict = self.training_theme_parts_with_color
        else:
            keyword = 'parts'
            theme_dict = self.training_theme_parts

        self.training_piece_idf = dict(self.get_piece_freq_document(exact_bool, False))
        
        scores = defaultdict(float)
        parts_dict = theme_dict[theme][keyword]
        for p in parts_dict:
            tf = parts_dict[p]
            doc_len = sum(parts_dict.values())
            N = len(self.training_theme_category)
            df = self.training_piece_idf[p]["in_themes_num"]
            scores[p] = self.tf_idf(tf, doc_len, N, df)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for p, score in sorted_scores[:n]:
            print("Piece: {}, TF-IDF: {}".format(p, round(score, 5)))

if __name__ == '__main__':
    lego = Lego()
    sets = lego.all_sets
    # Run one time below
    # lego.get_training_testing_set('training.json', 'testing.json')

    # lego.compare_two_sets('71006-1', '10220-1')
    # lego.compare_with_all('10220-1')
    # print sum(sets['10220-1']['parts_with_color'].values())
    # print sum(sets['10220-1']['parts'].values())
    # print sets['71006-1']['num_pieces_plus_spare']
    
    # set_id_args = ['71006-1', '10220-1']
    # print lego.merge_parts_dicts(set_id_args)
    # print lego.get_piece_freq(False)

    # a = lego.get_piece_freq(True, False)
    # b = lego.get_piece_freq(False, False)

    #c = lego.get_piece_freq_document(True, False)
    
    #d = lego.get_piece_freq_document(False, True)
    
    #lego.get_popular_piece_of_each_set(10, True)
    # lego.get_popular_piece_of_each_set(3, False)

    # lego.get_theme_piece(2, 50, "Friends", False)
    #lego.get_popular_piece_of_theme(10, "Friends", False)
    # lego.get_popular_piece_of_theme_new(10, "Friends", False)
    
    # lego.get_piece_categories()
    print lego.helper_stop_bricks(0.3)
    
    #lego.get_piece_categories_remove_stop_bricks(0.4, 6)
    # lego.get_all_parts(False)

    # lego.get_piece_companions(False)
    



