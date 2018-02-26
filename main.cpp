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
#include <pthread.h>
#include <unistd.h>
#include <iomanip>

using namespace std;

/* Precisions */
typedef unsigned long ulong;
typedef double real;

/* Item, item type, and behavior information */
ulong items_num, itypes_num, behaviors_num;
vector<ulong> items, itypes;
map<ulong, ulong> item2itype, item2item_idx, itype2itype_idx, item_idx2itype_idx;
vector<vector<ulong> > itype_idx2item_indices, behaviors;

/* Embeddings */
vector<vector<real> > item_embs;

/* Parameters */
ulong dim  = 128, threads_num = 4, total_samples = (long)1e6;
real rho = 0.025;

/* Misc */
ulong curr_samples = 0;
real curr_rho = rho;


/*
 * Initialize item embeddings
 */
void initialize_item_embs() {
    cout << "Initializing..." << endl;
    random_device rd;
    /* Random engines */
    mt19937 engine(rd());
    // knuth_b engine(rd());
    // default_random_engine engine(rd()) ;

    real low = -1e-3;
    real high = 1e-3;
    uniform_real_distribution<real> dist(low, high);
    for (int i=0; i<items_num; i++) {
        vector<real> item_emb;
        item_emb.reserve(dim);  // Avoid re-allocations and improve ~15% efficiency
        for (int d=0; d<dim; d++) {
            item_emb.push_back(dist(engine));
        }
        item_embs.push_back(item_emb);
        item_emb.clear();
    }
}


/*
 * Single thread for training LearnSUC model
 */
void *train_learn_suc_thread(void *threadid) {
    auto t_id = (long)threadid;
    auto checkpoint_interval = (ulong)1e3;
    ulong t_samples = 0, checkpoint_samples = 0;

    sleep((unsigned int)t_id + 1);
    //cout << "train_learn_suc_thread(): thread  " << t_id  << " running..." << endl;

    while (t_samples < total_samples/threads_num + 1) {
        // TODO: negative sampling, compute gradient, and update embeddings

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
    pthread_exit(NULL);
}


/*
 * Train LearnSUC model by multi-threading
 */
void train_learn_suc() {
    long t_id;
    pthread_t threads[threads_num];

    cout << "Start training LearnSUC model..." << endl;
    for (t_id = 0; t_id < threads_num; t_id++) {
        //cout << "train_learn_suc(): creating thread " << t_id << endl;
        pthread_create(&threads[t_id], NULL, train_learn_suc_thread, (void *)t_id);
    }

    for (t_id = 0; t_id < threads_num; t_id++) {
        pthread_join(threads[t_id], NULL);
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

    set<ulong> itype_set;
    /* Make items, item2itype, item2item_idx */
    ulong item_idx_count = 0;
    for (string line; getline(filein, line); ) {  // item_idx follow input items order
        vector<ulong> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim)) {
            tokens.push_back(stoul(token));  // Parse to ulong
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
    ulong itype_idx_count = 0;
    for (auto it=itype_set.begin(); it!=itype_set.end(); ++it) {
        itypes.push_back(*it);
        itype2itype_idx.insert(make_pair(*it, itype_idx_count));
        itype_idx_count ++;
    }
    itypes_num = itype2itype_idx.size();

    multimap<ulong, ulong> itype_idx2item_idx;
    /* Make item_idx2itype_idx */
    for (auto &it : item2itype) {
        item_idx2itype_idx.insert(make_pair(item2item_idx[it.first], itype2itype_idx[it.second]));
        itype_idx2item_idx.insert(make_pair(itype2itype_idx[it.second], item2item_idx[it.first]));
    }

    /* Make itype_idx2item_indices */
    for (auto it=itype_set.begin(); it!=itype_set.end(); ++it) {
        vector<ulong> item_indices;
        auto range = itype_idx2item_idx.equal_range(itype2itype_idx[*it]);
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
    for (int i=0; i<itype_idx2item_indices.size(); i++){
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
    set<ulong> items_set(items.begin(), items.end());
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
            vector<ulong> tokens2;
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

void output_embs (){
    cout << "Writing out item embeddings..." << endl;
    // TODO: implement output embeddings
    sleep(3);
    cout << "Done!" << endl;
}


int main() {
    using clock = chrono::steady_clock;

    // TODO: parse arguments from command line
    string itemlist_file = "data/itemlist.txt";
    string behaviorlist_file = "data/behaviorlist.txt";

    /* Read in itemlist, behaviorlist files and initialization */
    auto t_i_s = clock::now();
    read_itemlist_file(itemlist_file);
    read_behaviorlist_file(behaviorlist_file);
    initialize_item_embs();
    auto t_i_e = clock::now();
    cout << endl;

    /* Train LearnSUC use multi-threading */
    auto t_t_s = clock::now();
    train_learn_suc();
    auto t_t_e = clock::now();
    cout << endl;

    /* Output embedding */
    auto t_o_s = clock::now();
    output_embs();
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

    /* TEST TEST TEST TEST TEST */

    return 0;
}