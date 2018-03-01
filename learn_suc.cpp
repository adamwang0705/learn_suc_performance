/*
 *  This is the performance implementation of LearnSUC model.
 *
 *  @author Daheng Wang
 *  @email  dwang8@nd.edu
 *
 *  Publication: Multi-Type Itemset Embedding for Learning Behavior Success
 *  Authors: Daheng Wang, Meng Jiang, Qingkai Zeng, Zachary Eberhart, Nitesh Chawla
 *  Organization: University of Notre Dame, Notre Dame, Indiana, 46556, USA
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <algorithm>

#define MAX_STR_LEN 100

using namespace std;

/* Precisions */
typedef long lint;
typedef double real;

/* Item, item type, and behavior information */
lint items_num, itypes_num, behaviors_num;
vector<lint> items, itypes;
map<lint, lint> item2itype, item2item_idx, itype2itype_idx, item_idx2itype_idx;
vector<vector<lint>> itype_idx2item_indices, behaviors;
vector<real> itype_weights;

/* Embeddings */
vector<vector<real>> item_embs;

/* Parameters */
char itemlist_file[MAX_STR_LEN], behaviorlist_file[MAX_STR_LEN], output_file[MAX_STR_LEN];
lint dim  = 128, threads_num = 8, total_samples = (lint)1e3, negative = 10, mode = 1;
real rho = 0.025;

/* Misc */
lint curr_samples = 0;
real curr_rho = rho;


/*
 * Initialize item embeddings
 */
void initialize() {
    cout << "Initializing..." << endl;

    /* Item type weights */
    real default_weight  = 1.0;
    for (lint i=0; i<itypes_num; ++i) {
        itype_weights.push_back(default_weight);
    }

    /* Item embeddings */
    random_device rd;
    /* Random engines */
    mt19937 engine(rd());
    // knuth_b engine(rd());
    // default_random_engine engine(rd()) ;

    real low = -1e-3;
    real high = 1e-3;
    uniform_real_distribution<real> dist(low, high);
    for (lint i=0; i<items_num; i++) {
        vector<real> item_emb;
        item_emb.reserve(static_cast<unsigned long>(dim));  // Avoid re-allocations and improve ~15% efficiency
        for (lint d=0; d<dim; d++) {
            item_emb.push_back(dist(engine));
        }
        item_embs.push_back(item_emb);
        item_emb.clear();
    }
}

real quick_pos_sinh(real pos_b_norm) {
    if (pos_b_norm > 10) {
        return 0;
    } else if (pos_b_norm < 0.1) {
        return 100;
    } else {
        return 1/pos_b_norm/sinh(pos_b_norm);
    }
}

real quick_neg_sinh(real neg_b_norm) {
    if (neg_b_norm < 0.05 or neg_b_norm > 100) {
        return 0;
    } else {
        return 1/neg_b_norm/neg_b_norm/neg_b_norm/sinh(1/neg_b_norm);
    }
}


/*
 * Single thread for training LearnSUC model
 */
void *train_learn_suc_thread(void *) {
    //auto t_id = (long)threadid;
    auto checkpoint_interval = (lint)1e3;
    lint t_samples = 0, checkpoint_samples = 0;

    random_device rd;
    mt19937 engine(rd());

    while (t_samples < total_samples/threads_num + 1) {
        /* Positive behavior */
        // Sample a positive behavior index
        uniform_int_distribution<lint> dis_pos_b(0, behaviors_num);
        auto pos_b = dis_pos_b(engine);

        auto pos_b_size = behaviors[pos_b].size();
        vector<lint> pos_b_t_i_counts(static_cast<unsigned long>(itypes_num));

        // Compute pos behavior vector (type weighted sum of item vectors)
        vector<real> vec_pos_b(static_cast<unsigned long>(dim));
        for(const auto& i: behaviors[pos_b]) {
            real tw = itype_weights[item_idx2itype_idx[i]];
            pos_b_t_i_counts[item_idx2itype_idx[i]] += 1;
            for(lint d=0; d<dim; ++d) {
                vec_pos_b[d] += item_embs[i][d] * tw;
            }
        }

        // Compute pos behavior norm, pos_sinh
        real norm_pos_b = 0;
        for (lint d=0; d<dim; ++d) {
            norm_pos_b += pow(vec_pos_b[d], 2);
        }
        norm_pos_b = sqrt(norm_pos_b);
        auto pos_b_sinh = quick_pos_sinh(norm_pos_b);

        // Compute gradient
        vector<real> vec_pos_b_inc(static_cast<unsigned long>(dim));
        for(lint d=0; d<dim; ++d) {
            vec_pos_b_inc[d] += pos_b_sinh * vec_pos_b[d];
        }

        // Update item embeddings
        for(const auto& i: behaviors[pos_b]) {
            real tw = itype_weights[item_idx2itype_idx[i]];
            for(lint d=0; d<dim; ++d) {
                item_embs[i][d] += vec_pos_b_inc[d] * tw * curr_rho;
            }
        }

        /* Negative sampling */
        for (int n=0; n<negative; ++n) {  // For each negative behavior sampling
            // Sample a negative behavior
            set<lint> neg_b_items;
            unsigned long cum_neg_b_i_count = 0;

            if (mode == 1) {  // size-constrained
                // Generate samples number of each item type
                vector<lint> neg_b_t_cuts;
                uniform_int_distribution<lint> dis_pos_b_t_cut(0, pos_b_size);
                for (int t=0; t<itypes_num-1; ++t) {
                    auto t_cut = dis_pos_b_t_cut(engine);
                    neg_b_t_cuts.push_back(t_cut);
                }
                sort(neg_b_t_cuts.begin(), neg_b_t_cuts.end());

                vector<lint> neg_b_t_i_counts(static_cast<unsigned long>(itypes_num));
                for(int t=0; t<itypes_num; ++t) {
                    if (t == 0) {
                        neg_b_t_i_counts.push_back(neg_b_t_cuts[t]);
                    } else {
                        neg_b_t_i_counts.push_back(neg_b_t_cuts[t] - neg_b_t_cuts[t-1]);
                    }
                }
                neg_b_t_i_counts.push_back(pos_b_size - neg_b_t_cuts.back());

                for(int t=0; t<itypes_num; ++t) {   // For each item type
                    uniform_int_distribution<lint> dis_neg_b_t(0, itype_idx2item_indices[t].size());

                    auto t_i_count = neg_b_t_i_counts[t];
                    cum_neg_b_i_count += t_i_count;

                    // Sample same number of same type items as positive behavior (without duplicates)
                    while(neg_b_items.size() < cum_neg_b_i_count) {
                        auto neg_b_t_i = dis_neg_b_t(engine);
                        neg_b_items.insert(neg_b_t_i);
                    }
                }
            } else if (mode == 2) {  // type-distribution constrained
                for(int t=0; t<itypes_num; ++t) {   // For each item type
                    uniform_int_distribution<lint> dis_neg_b_t(0, itype_idx2item_indices[t].size());

                    auto t_i_count = pos_b_t_i_counts[t];
                    cum_neg_b_i_count += t_i_count;

                    // Sample same number of same type items as positive behavior (without duplicates)
                    while(neg_b_items.size() < cum_neg_b_i_count) {
                        auto neg_b_t_i = dis_neg_b_t(engine);
                        neg_b_items.insert(neg_b_t_i);
                    }
                }
            } else {
                cout << "Negative sampling mode wrong!" << endl;
                break;
            }

            // Compute neg behavior vector (type weighted sum of item vectors)
            vector<real> vec_neg_b(static_cast<unsigned long>(dim));
            for(const auto& i: neg_b_items) {
                real tw = itype_weights[item_idx2itype_idx[i]];
                for (lint d=0; d<dim; ++d) {
                    vec_neg_b[d] += item_embs[i][d] * tw ;
                }
            }

            // Compute neg behavior norm, pos_sinh
            real norm_neg_b = 0;
            for (lint d=0; d<dim; ++d) {
                norm_neg_b += pow(vec_neg_b[d], 2);
            }
            norm_neg_b = sqrt(norm_neg_b);
            auto neg_b_sinh = quick_neg_sinh(norm_neg_b);

            // Compute gradient
            vector<real> vec_neg_b_inc(static_cast<unsigned long>(dim));
            for(lint d=0; d<dim; ++d) {
                vec_neg_b_inc[d] += neg_b_sinh * vec_neg_b[d];
            }

            // Update item embeddings
            for(const auto& i: neg_b_items) {
                real tw = itype_weights[item_idx2itype_idx[i]];
                for(lint d=0; d<dim; ++d) {
                    item_embs[i][d] -= vec_neg_b_inc[d] * tw * curr_rho;
                }
            }
        }

        /* Check for checkpoint */
        if (t_samples - checkpoint_samples >= checkpoint_interval) {
            /* Update checkpoint */
            curr_samples += t_samples - checkpoint_samples;
            checkpoint_samples = t_samples;

            /* Report progress */
            cout << fixed << setw(10) << setprecision(6)
                 << "Current learning rate: " << curr_rho << "; "
                 << fixed << setw(5) << setprecision(2)
                 << "Progress: " << (real)curr_samples/total_samples*100 << "%\r";
            cout.flush();

            /* Update learning rate */
            if (curr_rho < rho * 0.0001) {
                // Set minial learning rate
                curr_rho = rho * 0.0001;
            } else {
                curr_rho = rho * (1 - ((real)curr_samples/total_samples));
            }
        }

        /* Add thread samples counter */
        t_samples ++;
    }
    pthread_exit(nullptr);
}


/*
 * Train LearnSUC model by multi-threading
 */
void train_learn_suc() {
    long t_id;
    pthread_t threads[threads_num];

    cout << "Start training LearnSUC model..." << endl;
    for (t_id = 0; t_id < threads_num; t_id++) {
        //pthread_create(&threads[t_id], nullptr, train_learn_suc_thread, (void *)t_id);
        pthread_create(&threads[t_id], nullptr, train_learn_suc_thread, nullptr);
    }

    for (t_id = 0; t_id < threads_num; t_id++) {
        pthread_join(threads[t_id], nullptr);
    }
    cout << endl << "Done!" << endl;
}


/*
 * Read in item information from itemlist file.
 * Each line follows format: <non_neg_int_item>\t<non_neg_int_item_type>
 */
void read_itemlist_file(const string &itemlist_file, char delim='\t') {
    cout << "Input itemlist file: " << itemlist_file << endl;
    ifstream filein(itemlist_file);

    set<lint> itype_set;
    /* Make items, item2itype, item2item_idx */
    lint item_idx_count = 0;
    for (string line; getline(filein, line); ) {  // item_idx follow input items order
        vector<lint> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim)) {
            tokens.push_back(stol(token));  // Parse to lint
        }
        if(tokens.size() != 2) {  // Line format wrong
            cout << "Input itemlist file format wrong!" << endl;
            break;
        } else {
            items.push_back(tokens[0]);
            item2itype.insert(make_pair(tokens[0], tokens[1]));
            item2item_idx.insert(make_pair(tokens[0], item_idx_count));
            item_idx_count ++;
            itype_set.insert(tokens[1]);
        }
    }
    items_num = item2itype.size();

    cout << "Indexing item types..."<< endl;
    /* Make itypes, itype2itype_idx */
    lint itype_idx_count = 0;
    for (const auto& it: itype_set) {
        itypes.push_back(it);
        itype2itype_idx.insert(make_pair(it, itype_idx_count));
        itype_idx_count ++;
    }
    itypes_num = itype2itype_idx.size();

    multimap<lint, lint> itype_idx2item_idx;
    /* Make item_idx2itype_idx */
    for (const auto& it : item2itype) {
        item_idx2itype_idx.insert(make_pair(item2item_idx[it.first], itype2itype_idx[it.second]));
        itype_idx2item_idx.insert(make_pair(itype2itype_idx[it.second], item2item_idx[it.first]));
    }

    /* Make itype_idx2item_indices */
    for (const auto& it: itype_set) {
        vector<lint> item_indices;
        auto range = itype_idx2item_idx.equal_range(itype2itype_idx[it]);
        for (auto i = range.first; i!=range.second; ++i) {
            item_indices.push_back(i->second);
        }
        itype_idx2item_indices.push_back(item_indices);
        item_indices.clear();
    }

    itype_set.clear();
    itype_idx2item_idx.clear();
    cout << "Done!\t" << "#items: " << items_num << "; #item types: " << itypes_num
         << "; Distribution (type-#): ";
    for (unsigned int i=0; i<itype_idx2item_indices.size(); i++){
        cout << itypes[i] << "-" << itype_idx2item_indices[i].size() << " ";
    }
    cout << endl;
}

/*
 * Read in behaviors information from behaviorlist file.
 * Each line follows format: <non_neg_int_behavior>\t[<non_neg_int_item1>,<non_neg_int_item2>,...]
 */

void read_behaviorlist_file(const string &behaviorlist_file, char delim1='\t', char delim2=',') {
    cout << "Input behaviorlist file: " << behaviorlist_file << endl;
    ifstream filein(behaviorlist_file);

    /* Make behaviors */
    set<lint> items_set(items.begin(), items.end());
    bool in_items_set;
    for (string line; getline(filein, line); ) {
        vector<string> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim1)) {
            tokens.push_back(token);
        }
        if(tokens.size() != 2) {  // Line format wrong
            cout << "Input behaviorlist file format wrong!" << endl;
            break;
        } else {
            vector<lint> tokens2;
            stringstream ss2(tokens[1]);  // tokens[0] for behavior; tokens[1] for items
            string token2;
            while (getline(ss2, token2, delim2)) {
                /* Discard items not in itemlist file */
                in_items_set = items_set.find(stoul(token2)) != items_set.end();
                if(in_items_set) {
                    tokens2.push_back(item2item_idx[stoul(token2)]);
                }
            }
            behaviors.push_back(tokens2);
        }
    }
    behaviors_num = behaviors.size();

    items_set.clear();
    cout << "Done!\t" << "#behaviors: " << behaviors_num << endl;
}

void output_embs (const string &output_file){
    cout << "Writing out item embeddings..." << endl;
    ofstream fileout(output_file);
    /* Write header line */
    fileout << items_num << "\t" << dim << endl;
    /* Write embeddings */
    for (int i=0; i<items_num; ++i) {
        fileout << items[i] << "\t";
        for (int d=0; d<dim-1; ++d) {
            fileout << item_embs[i][d] << "\t";
        }
        fileout << item_embs[i].back() << endl;
    }
    cout << "Done!" << endl;
}

int parse_args(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) {
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                cout << "Argument missing for " << str << endl;
                exit(1);
            }
            return a;
        }
    }
    return -1;
}


int main(int argc, char **argv) {
    /* Parse arguments */
    int i;
    if ((i = parse_args((char *)"--itemlist", argc, argv)) > 0) strcpy(itemlist_file, argv[i + 1]);
    if ((i = parse_args((char *)"--behaviorlist", argc, argv)) > 0) strcpy(behaviorlist_file, argv[i + 1]);
    if ((i = parse_args((char *)"--output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);

    if ((i = parse_args((char *)"--size", argc, argv)) > 0) dim = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--mode", argc, argv)) > 0) mode = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--samples", argc, argv)) > 0) total_samples = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--negative", argc, argv)) > 0) negative = stol(argv[i + 1]);
    if ((i = parse_args((char *)"--rho", argc, argv)) > 0) rho = stod(argv[i + 1]);
    if ((i = parse_args((char *)"--threads", argc, argv)) > 0) threads_num = stol(argv[i + 1]);
    total_samples *= 1000;

    using clock = chrono::steady_clock;
    /* Read in itemlist, behaviorlist files and initialization */
    auto t_i_s = clock::now();
    read_itemlist_file(itemlist_file);
    read_behaviorlist_file(behaviorlist_file);
    initialize();
    auto t_i_e = clock::now();
    cout << endl;

    /* Train LearnSUC use multi-threading */
    auto t_t_s = clock::now();
    train_learn_suc();
    auto t_t_e = clock::now();
    cout << endl;

    /* Output embedding */
    auto t_o_s = clock::now();
    output_embs(output_file);
    auto t_o_e = clock::now();
    cout << endl;

    /* Print summary */
    auto init_time = (real)chrono::duration_cast<chrono::seconds>(t_i_e-t_i_s).count();
    auto training_time = (real)chrono::duration_cast<chrono::seconds>(t_t_e-t_t_s).count();
    auto output_time = (real)chrono::duration_cast<chrono::seconds>(t_o_e-t_o_s).count();
    auto total_time = (real)chrono::duration_cast<chrono::seconds>(t_o_e-t_i_s).count();
    cout << fixed << setw(5) << setprecision(2) << "Total elapsed time: " << total_time << " s" << endl
         << " - Initialization: " << init_time << " s"
            << " (" << (init_time/total_time)*100 << "%)" << endl
         << " - Training: " << training_time << " s"
            << " (" << (training_time/total_time)*100 << "%)" << endl
         << " - Output: " << output_time << " s"
            << " (" << (output_time/total_time)*100 << "%)";
    cout << endl;

    return 0;
}