
#include <map>
#include <bitset>
#include <iostream>

#include "msl/fwcounter.h"
#include "ooo_cpu.h"

// Architecture is from the paper below
// https://people.eecs.berkeley.edu/~kubitron/courses/cs558-S04/handouts/papers/p3-lee.pdf
// We split the second level table into two halves 
namespace
{
constexpr std::size_t TABLE_SIZE = 32768;
constexpr std::size_t COUNTER_BITS = 3;
constexpr std::size_t TAG_SIZE = 8;
constexpr std::size_t GLOBAL_HISTORY_LENGTH = 15; // How long the history register is 
constexpr std::size_t tag_mask = 0xffffffffffffff00;

struct YAGs
    {
        std::bitset<TAG_SIZE> tag;
        champsim::msl::fwcounter<COUNTER_BITS> counter;
    };
    
int count[4] = {0,0,0,0};
std::bitset<GLOBAL_HISTORY_LENGTH > GLOBAL_HISTORY;

std::map<O3_CPU*, std::array<champsim::msl::fwcounter<COUNTER_BITS>, TABLE_SIZE>> choice_predictor;
std::map<O3_CPU*, std::array<YAGs, TABLE_SIZE>> Taken_table;
std::map<O3_CPU*, std::array<YAGs, TABLE_SIZE>> Not_Taken_table;
} // namespace

void O3_CPU::initialize_branch_predictor() {}

uint8_t O3_CPU::predict_branch(uint64_t ip)
{

    champsim::msl::fwcounter<COUNTER_BITS> value;
    // We start by hashing the choice predictor with the ip
    auto hash = ip % ::TABLE_SIZE;
    auto choice = ::choice_predictor[this][hash];

    std::bitset<TAG_SIZE> tag_value =  tag_mask ^ ip;
    int tcache = (tag_value == ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag);
    int ntcache = (tag_value == ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag);

    if (choice.value() >= (choice.maximum / 2)) {
        if (ntcache) {
            count[0]++;
            value = ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].counter;
            return (value.value() >= (value.maximum / 2));
        }
        else {
            count[1]++;
            return (choice.value() >= (choice.maximum / 2));
        }
    }
    else {
        if (tcache) {
            count[2]++;
            value = ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].counter;
            return (value.value() >= (value.maximum / 2));
        }
        else {
            count[3]++;
            return (choice.value() >= (choice.maximum / 2));
        }
    }
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{

    // if(taken)
    //     std::cout << "IP" << ip << "Branch_target" << branch_target << std::endl;
    
    // Update the choice predictor 
    auto hash = ip % ::TABLE_SIZE;
    auto choice = ::choice_predictor[this][hash];


    champsim::msl::fwcounter<COUNTER_BITS> value;

    std::bitset<TAG_SIZE> tag_value = tag_mask ^ ip;
    int tcache = (tag_value == ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag);
    int ntcache = (tag_value == ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag);

    if (choice.value() >= (choice.maximum / 2))
    {
        if (!taken) {
            ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag = tag_value;
            ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].counter += -1;
        }
        if (ntcache) {
            ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag = tag_value;
            ::Not_Taken_table[this][GLOBAL_HISTORY.to_ullong()].counter += taken ? 1 : -1;
        }
        else {
            ::choice_predictor[this][hash] += taken ? 1 : -1;   
        }

    }
    else
    {
        if (taken){
            ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag = tag_value;
            ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].counter  += 1;
        }
        if (tcache) {
            ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].tag = tag_value;
            ::Taken_table[this][GLOBAL_HISTORY.to_ullong()].counter  += taken ? 1 : -1;
        }
        else {
            ::choice_predictor[this][hash] += taken ? 1 : -1;   
        }
    }

                

 
    GLOBAL_HISTORY >>= 1; // Shift the history register to the left to remove the least recent data 
    GLOBAL_HISTORY[GLOBAL_HISTORY_LENGTH-1] = taken; // insert the most recent data into the history register 
}
