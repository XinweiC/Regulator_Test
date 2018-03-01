#include<iostream>
#include<fstream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include<vector>
#include<algorithm>
#include<thread>
#include <sys/ptrace.h>
#include <sys/time.h>
#include <sys/user.h>
#include <sys/wait.h>
#include<map>
#include <armadillo>
#include "papi.h"
using namespace std;
using namespace arma;
namespace{
    volatile bool is_timer_done = false;
}

#define TARGET_THROUGHPUT 2750
//#define TARGET_POWER 6               //wendy power regulation 
long SamplePeriodUsecs = 10000; //1000;

// New constants that I added. -Yiqun
float TARGET_POWER = 6;
float a = 0.190654; // default value for a and b
float b = 4.15;
float c = 0.2;
float d = 1;
float target;

float ConstantFreq = 3400000;
//wendy
/*truct SysIdent {
    float f1;
    float f2;
    float f3;
    float pow;
};
std::vector<SysIdent> SysParameter;
fmat H=ones<fmat>(1,4);*/
fmat X_si=ones<fmat>(1,4);
//fmat P=ones<fmat>(1,1);
fmat ky = ones<fmat>(4,4);   //wendy identity inital matrix for system identification
//H<<1<<1<<1<<1<<endr;
//std::cout<<"initial H= "<<H<<std::endl<<std::flush;
//P<<1<<endr;
//std::cout<<"initial P= "<<P<<std::endl<<std::flush;
//wendy
const long kMicroToBase = 1e6;
struct stats_t{
    long time;
    vector<long long>results;
    long curr_fq;
    long next_fq;
};

struct event_info_t{
  int component;
  int set;
  std::vector<int> codes;
  std::vector<std::string> names;
  char unit [PAPI_MIN_STR_LEN];
};

std::vector<event_info_t> init_papi_counters(
    const std::vector<std::string>& event_names) {
  std::vector<event_info_t> eventsets;
  int retval;
  for (auto& event_name : event_names) {
    int event_code;
    retval = PAPI_event_name_to_code(const_cast<char*>(event_name.c_str()), 
                                     &event_code);
    if (retval != PAPI_OK) {
      std::cerr << "Error: bad PAPI event name \"" << event_name << "\" to code: ";
      PAPI_perror(NULL);
      exit(-1);
    }
    int component = PAPI_get_event_component(event_code);
    auto elem = find_if(
        begin(eventsets), end(eventsets),
        [&](const event_info_t& c) { return c.component == component; });
    if (elem == end(eventsets)) {
      event_info_t new_info;
      new_info.component = component;
      new_info.codes.push_back(event_code);
      new_info.names.emplace_back(event_name);
      eventsets.push_back(new_info);
    } else {
      elem->codes.push_back(event_code);
      elem->names.emplace_back(event_name);
    }
  }

  for (auto& event : eventsets) {
    int eventset = PAPI_NULL;
    retval = PAPI_create_eventset(&eventset);
    if (retval != PAPI_OK) {
      std::cerr << "Error: bad PAPI create eventset: ";
      PAPI_perror(NULL);
      exit(-1);
    }
    retval = PAPI_add_events(eventset, &event.codes[0], event.codes.size());
    if (retval != PAPI_OK) {
      std::cerr << "Error: bad PAPI add eventset: ";
      PAPI_perror(NULL);
      exit(-1);
    }
    event.set = eventset;
  }
  return eventsets;
}


void start_counters(const std::vector<event_info_t>& counters){
  for(auto& counter : counters){
    auto retval = PAPI_start(counter.set);
    if (retval != PAPI_OK) {
      std::cerr << "Error: bad PAPI start eventset: ";
      PAPI_perror(NULL);
      exit(-1);
    }
  }
}

std::vector<std::vector<long long>> stop_counters(const std::vector<event_info_t>& counters){
  std::vector<std::vector<long long>> results;
  for(const auto& counter : counters){
    std::vector<long long> counter_results(counter.codes.size());
    auto retval = PAPI_stop(counter.set, &counter_results[0]);
    if(retval != PAPI_OK){
      std::cerr << "Error: bad PAPI stop eventset: ";
      PAPI_perror(NULL);
      exit(-1);
    }
    results.push_back(counter_results);
  }
  return results;
}

void attach_counters_to_core(const std::vector<event_info_t>& counters, int cpu_num) {
  for(auto& counter : counters){
    PAPI_option_t options;
    options.cpu.eventset = counter.set;
    options.cpu.cpu_num = cpu_num;
    int retval = PAPI_set_opt(PAPI_CPU_ATTACH, &options);
    if(retval != PAPI_OK) {
      std::cerr << "Error: unable to CPU_ATTACH core " << cpu_num << ": ";
      PAPI_perror(NULL);
      exit(-1);
    }
  }
}

long getFrequency(int cpu_id){
	char ffile[60];
	sprintf(ffile, 
					"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", 
					cpu_id); 
	FILE* fp  = fopen(ffile, "r");
	if(!fp){
		fprintf(stderr, "Error: Unable to open cpufreq file %s\n", ffile);
		return 0;
	}
	long res;
	fscanf(fp, "%ld", &res);
	fclose(fp);
	return res;

}

int setFrequency(int cpu_id, long freq){
	char ffile[60];
    int i=cpu_id;
	while(getFrequency(cpu_id) != freq){
		while( i==cpu_id ){
//            printf("entred\n");
			sprintf(ffile, 
							"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed", 
							i); 
			FILE* fp  = fopen(ffile, "w");
			if(!fp){
				fprintf(stderr, "Error: Unable to open cpufreq file %s\n", ffile);
				return 0;
			}
			fprintf(fp, "%ld", freq);
			fclose(fp);
            i++;
		}
	}
	return 1;
}

// The logic in the control loop is changed to be the cubic model. -Yiqun
float basic_control_loop(float last_power, long freq)
{
    double err = target - last_power;
   // double dev = 3 * a * freq * freq + b;
   //wendy
/* LS method but singular inverse trouble
	std::cout<<"inbasicl control loop H= "<<H<<" P= "<<P<<std::endl;
	std::cout<<"H transpose= "<<H.t()<<std::endl;
	std::cout<<"inverser of H transpose times H=  "<<inv(H*H.t())<<std::endl;
	std::cout<<"H transpose times P "<<(H.t()*P)<<std::endl;
	fmat left = inv (H.t()*H);
	fmat right = H.t()*P;
	//X_si = (inv(H.t()*H))*(H.t()*P);
	std::cout<<"left= "<<left<<"right= "<<right<<std::endl;
	X_si = left * right;
	std::cout<<"X_si= "<<X_si<<std::endl<<std::flush;

	std::cout<<"previous ky= "<<ky<<std::endl;
	std::cout<<"incontrolloop temp_H= "<<H<<std::endl;
	std::cout<<"incontrolloop temp_p== "<<P<<std::endl;
 	std::cout<<"fenzi= "<<ky*H.t()*H*ky<<std::endl;
	std::cout<<"fenmu= "<<(1+H*ky*H.t())<<std::endl;
	fmat aaa = ky*H.t()*H*ky;
	std::cout<<"aaa= "<<aaa<<std::endl;
	fmat bbb = 1+H*ky*H.t();
	std::cout<<"bbb= "<<bbb<<endl;
	float cc= 1/det(bbb);
	std::cout<<"cc= "<<cc<<std::endl;
	fmat ddd = aaa*cc;
	std::cout<<"ddd= "<<ddd<<std::endl;
	ky = ky - (ky*H.t()*H*ky)/(1+H*ky*H.t());
	std::cout<<"new ky= "<<ky<<std::endl;
	fmat t_X_si = X_si.t() - ky*H.t()*(H*X_si.t()-P);
	X_si = t_X_si.t();
*/	//std::cout<<"new ky= "<<ky<<std::endl;
	//std::cout<<"previous ky= "<<ky<<std::endl;
	fmat temp_H, temp_P;
	float tt_freq = freq/1e6;
	temp_H<<(tt_freq * tt_freq *tt_freq)<<(tt_freq * tt_freq)<<tt_freq<<1<<endr;	
	//std::cout<<"temp_H= "<<temp_H<<std::endl;
	temp_P<<last_power<<endr;
	//std::cout<<"temp_P= "<<temp_P<<std::endl;
	fmat tt_ky;
	tt_ky = ky - (ky*temp_H.t()*temp_H*ky)/det(1+(temp_H*ky*temp_H.t()));
	ky = tt_ky;
	//std::cout<<"ky= "<<ky<<std::endl;
	fmat t_X_si = X_si.t() - ky*temp_H.t()*(temp_H*X_si.t() - temp_P);
	X_si = t_X_si.t();
	//std::cout<<"X_si= "<<X_si<<std::endl;
	a = X_si(0)*1e-18;
	b = X_si(1)*1e-12;
	c = X_si(2)*1e-6;
	d = X_si(3);
	//std::cout<<"in control loop freq="<<freq<<std::endl;
	float dev = 3*a*freq*freq + 2*b*freq + c;
	//std::cout<<" a= "<<a<<" b= "<<b<<" c= "<<c<<" d= "<<d<<std::endl<<std::flush;
	//std::cout<<"dev="<<dev<<std::endl<<std::flush;
	//std::cout<<"err= "<<err<<std::endl;
	//std::cout<<"last_power= "<<last_power<<std::endl;
	float new_freq = freq + err/dev;
        //std::cout<<"err/dev="<<(err/dev)<<std::endl<<std::flush;
        if(isnan(new_freq))//maintain the current frequency if the core in not used
            {
                new_freq=freq;
            }
    
    return new_freq;
}


long select_proper_freq(float frequency_exact)
{
    long new_freq;

    if(frequency_exact>=3300000)
        new_freq=3400000;
    if(3150000<=frequency_exact && frequency_exact<3300000)
        new_freq=3200000;
    if(3000000<=frequency_exact && frequency_exact<3150000)
        new_freq=3100000;
    if(2800000<=frequency_exact && frequency_exact<3000000)
        new_freq=2900000;
    if(2600000<=frequency_exact && frequency_exact<2800000)
        new_freq=2700000;
    if(2450000<=frequency_exact && frequency_exact<2600000)
        new_freq=2500000;
    if(2300000<=frequency_exact && frequency_exact<2450000)
        new_freq=2400000;
    if(2100000<=frequency_exact && frequency_exact<2300000)
        new_freq=2200000;
    if(1900000<=frequency_exact && frequency_exact<2100000)
        new_freq=2000000;
    if(1750000<=frequency_exact && frequency_exact<1900000)
        new_freq=1800000;
    if(1600000<=frequency_exact && frequency_exact<1750000)
        new_freq=1700000;
    if(1400000<=frequency_exact && frequency_exact<1600000)
        new_freq=1500000;
    if(1200000<=frequency_exact && frequency_exact<1400000)
        new_freq=1300000;
    if(1050000<=frequency_exact && frequency_exact<1200000)
        new_freq=1100000;
    if(900000<=frequency_exact && frequency_exact<1050000)
        new_freq=1000000;
    if(frequency_exact<900000)
        new_freq=800000;

   return new_freq;
// return ConstantFreq;
}




void overflow(int signum, siginfo_t* info, void* context){
  (void)info;
  (void)context;
  if(signum == SIGALRM){
    is_timer_done = true;
  }
}




void do_profiling(int profilee_pid,const long period)
{
      auto ncores = thread::hardware_concurrency();
      vector<string>ctr_names;
      vector<vector<event_info_t>>events_info_per_core;

      // variables added for power monitoring. -Yiqun
      vector<string> ctr_names_global;
      vector<event_info_t> events_info_global;

      vector<int> children_pids;
      events_info_per_core.reserve(ncores);


      /* Initialize PAPI*/       
      printf("Init PAPI\n");
      int retval;
      if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
        cerr << "Unable to init PAPI library - " << PAPI_strerror(retval) << endl;
        exit(-1);
      }

      ctr_names.push_back("PAPI_TOT_INS");//event for total instructions 
      ctr_names.push_back("PAPI_TOT_CYC");//event for total cycles
      ctr_names_global.push_back("rapl:::PP0_ENERGY:PACKAGE0"); // event for energy -Yiqun

      // setup all global counters (energy counters) -Yiqun
      events_info_global=init_papi_counters(ctr_names_global);
      auto& global_counters = events_info_global;
      start_counters(global_counters);

      // setup all per core counters
      for(unsigned int i = 0; i < ncores; ++i){
        printf("Creating per-core counters on core %d\n", i);
        events_info_per_core.emplace_back(init_papi_counters(ctr_names));
        auto& counters = events_info_per_core[i];
        attach_counters_to_core(counters, i);
        start_counters(counters);
      }

      
      /*Setup tracing of all profilee threads*/
      children_pids.push_back(profilee_pid);

      
      //setup the timer//
      struct sigaction sa;
      sa.sa_sigaction = overflow;
      sa.sa_flags = SA_SIGINFO;
      if (sigaction(SIGALRM, &sa, nullptr) != 0) {
        cerr << "Unable to set up signal handler\n";
        exit(-1);
      }
      struct itimerval work_time;
      time_t sleep_secs = period / kMicroToBase;
      suseconds_t sleep_usecs = period % kMicroToBase; 
      work_time.it_value.tv_sec = sleep_secs;
      work_time.it_value.tv_usec = sleep_usecs;
      work_time.it_interval.tv_sec = sleep_secs;
      work_time.it_interval.tv_usec = sleep_usecs;
      setitimer(ITIMER_REAL, &work_time, nullptr);


      
      // Let the profilee run, periodically interrupting to collect profile data.
      ptrace(PTRACE_CONT, profilee_pid, nullptr, nullptr); // Allow child to fork
      int status;
      wait(&status); // wait for child to begin executing
      printf("Start profiling.\n");
      /* Reassert that we want the profilee to stop when it clones */
      ptrace(PTRACE_SETOPTIONS, profilee_pid, nullptr,
             PTRACE_O_EXITKILL | PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXIT);
      ptrace(PTRACE_CONT, profilee_pid, nullptr, nullptr); // Allow child to run!

      using core_id_t = int;
      using assignments_left_t = int;
      map<int, pair<core_id_t, assignments_left_t>> children_cores;

      auto start_time = PAPI_get_real_usec();
      
      for (;;) {
            auto wait_res= waitpid(-1, &status, __WALL); 

      if(wait_res == -1){ // bad wait!
      if(errno == EINTR && is_timer_done){ // timer expired, do profiling
            // halt timer
            work_time.it_value.tv_sec = 0;
            work_time.it_value.tv_usec = 0;
            setitimer(ITIMER_REAL, &work_time, nullptr);
            
            // kill all the children
            for(const auto& child : children_pids){
              kill(child, SIGSTOP);
            } 

      ////////collecting stats///////////
       // cerr<<"timer_expired"<<endl;
        vector<stats_t> statistics(ncores);
      
        auto cur_time = PAPI_get_real_usec();
        auto elapsed_time = cur_time - start_time;
        start_time = cur_time;

        vector<long long> global_results;

        int ncounters = 0;
        for(const auto& e : events_info_global){
            ncounters += e.codes.size();
          }
        global_results.resize(ncounters);
        int cntr_offset = 0;
        for(const auto& eventset : events_info_global  )
        {
            int retval=PAPI_stop(eventset.set,&global_results[cntr_offset]);
            if(retval != PAPI_OK){
              cerr << "Error: bad PAPI stop: ";
              PAPI_perror(NULL);
              terminate();
            }
            retval = PAPI_start(eventset.set);
            if(retval != PAPI_OK){
              cerr << "Error: bad PAPI stop: ";
              PAPI_perror(NULL);
              terminate();
            }
         cntr_offset += eventset.codes.size(); 
        }

        // collecting energy statistics. -Yiqun
        float total_energy=global_results[0] / 1e9;
        float power = total_energy / elapsed_time *1e6;
        std::cout<<"_____energy statistics"<<"_____:"<<std::endl<<std::flush;
        //cerr<<"unit: " << unit << "\n";
        std::cout<<"Energy used by all cores in = "<<total_energy<<" J"<<std::endl<<std::flush;
        std::cout<<"Average power = "<<power<<" W"<<std::endl<<std::flush;

	//wendy
	//SysIdent si;	
	//si.pow = power;
	//wendy

	
        for(unsigned int i = 0; i < ncores; ++i){
            vector<event_info_t>eventsets_vect=events_info_per_core[i];
            stats_t res;
            int ncounters = 0;
            for(const auto& e : eventsets_vect){
                ncounters += e.codes.size();
              }
            res.results.resize(ncounters);
            int cntr_offset = 0;
            for(const auto& eventset : eventsets_vect  )
            {
                int retval=PAPI_stop(eventset.set,&res.results[cntr_offset]);
                if(retval != PAPI_OK){
                  cerr << "Error: bad PAPI stop: ";
                  PAPI_perror(NULL);
                  terminate();
                }
                retval = PAPI_start(eventset.set);
                if(retval != PAPI_OK){
                  cerr << "Error: bad PAPI stop: ";
                  PAPI_perror(NULL);
                  terminate();
                }
             cntr_offset += eventset.codes.size(); 
            }
        
        res.time=elapsed_time;
        res.curr_fq=getFrequency(i);
        float total_instr=res.results[0]; 
        float total_cycles=res.results[1]; 
        float last_mips=float((total_instr)/(total_cycles)*(res.curr_fq))/float(1000);//the throughput in this cycle

        //cerr<<"_____statistics for core"<<i<<"_____:\n";
       // cerr<<"MIPS value = "<<last_mips<<"\n";
//wendy
/*	si.f1 = res.curr_fq;
	si.f2 = si.f1 * si.f1;
	si.f3 = si.f1 * si.f2;
	SysParameter.push_back (si);
	std::cout<<"Si.f1= "<<si.f1<<"si.f2= "<<si.f2<<std::endl;
	std::cout<<"previous H= "<<H<<std::endl;
	std::cout<<"previous P= "<<P<<std::endl;
	
	fmat temp_f, temp_p;
	temp_f<<(si.f1)<<(si.f2)<<(si.f3)<<1<<endr;
	std::cout<<"temp_f= "<<temp_f<<std::endl;

	temp_p<<power<<endr;
	std::cout<<"temp_p= "<<temp_p<<std::endl;
	H.insert_rows(H.n_rows,temp_f);
	P.insert_rows(P.n_rows,temp_p);
	std::cout<<"H= "<<H<<std::endl<<std::flush;
	std::cout<<"P= "<<P<<std::endl<<std::flush;*/
       float fq=basic_control_loop(power, res.curr_fq);//we run the throughput control algorithm
//wendy
        res.next_fq=select_proper_freq(fq);//select a particular frequency
        statistics[i]=res;
	if (i==1){
        std::cout<<"curr freq=\t"<<statistics[i].curr_fq<<std::endl<<std::flush;
        std::cout<<"next freq=\t"<<statistics[i].next_fq<<std::endl<<std::flush;
         }
	 //cerr<<"time elapsed\t"<<statistics[i].time<<"\n";
        
        }

        
        ///set new_frequency in all cores/////////
            for(int k=0;k<ncores;k++)
            {

                    setFrequency(k,statistics[k].next_fq);//setting freq for next control cycle
            }

        //////////////////////////////////////////////////////////////


        // read all the children registers
        for(const auto& child : children_pids){
          struct user_regs_struct regs;
          ptrace(PTRACE_GETREGS, child, nullptr, &regs);
          void* rip = (void*)regs.rip;
          auto child_core = children_cores[child].first;
        }

        // resume all children
        for(const auto& child : children_pids){
          ptrace(PTRACE_CONT, child, nullptr, nullptr);
        }
        is_timer_done = false;

        // resume timer
        work_time.it_value.tv_sec = sleep_secs;
        work_time.it_value.tv_usec = sleep_usecs;
        setitimer(ITIMER_REAL, &work_time, nullptr);
      } 
        else {
        cerr << "Error: unexpected return from wait - " << strerror(errno) << "\n";
        exit(-1);
        }
    }
      else{// good wait, add new thread
            if(status>>8 == (SIGTRAP | (PTRACE_EVENT_CLONE<<8))) { // new thread created
            printf("New thread created.\n");
            unsigned long new_pid;
            ptrace(PTRACE_GETEVENTMSG, wait_res, nullptr, &new_pid);
            auto pid_iter = find(begin(children_pids), end(children_pids), new_pid);
            if(pid_iter != end(children_pids)) {
              cerr << "Already have this newly cloned pid: " << new_pid << ".\n";
              exit(-1);
            }
            printf("Thread ID %lu created from thread ID %d\n", new_pid, wait_res);
            children_pids.push_back(new_pid);
            ptrace(PTRACE_SETOPTIONS, new_pid, nullptr,
                   PTRACE_O_EXITKILL | PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXIT);
            ptrace(PTRACE_CONT, wait_res, nullptr, nullptr);
          } else {
            if(status>>8 == (SIGTRAP | (PTRACE_EVENT_EXIT<<8))){
              printf("Deleting child %d\n", wait_res);
              auto pid_iter = find(begin(children_pids), end(children_pids), wait_res);
              if(pid_iter == end(children_pids)){
                cerr << "Error: Saw exit from pid " << wait_res << ". We haven't seen before!\n";
                exit(-1);
              }
              children_pids.erase(pid_iter);
              if(children_pids.size() == 0){ // All done, not tracking any more threads
                break;
              }
              printf("%lu children left\n", children_pids.size());
            }
            // always let the stopped tracee continue
            ptrace(PTRACE_CONT, wait_res, nullptr, nullptr);
          }
      
      }

    } 


}


int main(int argc, char* argv[], char* envp[]) {

 // if (sched_setaffinity(0, sizeof(mask), &mask)<0)     //assigen to one core.
//   printf("Error");

  // added code for extra input. -Yiqun
/* if (argc >= 3) {
    TARGET_POWER = atof(argv[2]);
  }
  if (argc >= 4) {
    SamplePeriodUsecs = atoi(argv[3]);
  }
  if (argc >= 5) {
    a = atof(argv[4]);
  }
  if (argc >= 6) {
    b = atof(argv[5]);
  }
*/
    if (argc >=3){
       TARGET_POWER = atof (argv[2]);
    }
    if  (argc>= 4){
       SamplePeriodUsecs = atoi (argv [3]);
    }

  // added print commands for debug purposes -Yiqun
  target = TARGET_POWER;
  std::cout<<"Target_power="<<TARGET_POWER<<"\t"<<"control cycle="<<SamplePeriodUsecs<<std::endl<<std::flush;
 // cerr<<"a = " << a << "\t" <<"b = " << b <<endl;
 // std::cout<<"ConstantFrequency="<< ConstantFreq << endl<<flush;
  auto period = SamplePeriodUsecs;
  
  if(argc==1){
      cerr<<"ERROR: benchmark executable name not specified\n";
  return 0;
  }


  // Fork a process to run the profiled application
  auto profilee = fork();
  if(profilee > 0){ /* parent */
    // Let's do this.
    ptrace(PTRACE_SETOPTIONS, profilee, nullptr,
           PTRACE_O_EXITKILL | PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXIT);
    do_profiling(profilee,period);
  } else if(profilee == 0){ /* profilee */
    // prepare for tracing
    ptrace(PTRACE_TRACEME, 0, nullptr, nullptr);
    raise(SIGSTOP);
    // start up client program
    execve(argv[optind], &argv[optind], envp); ///filename to execute
    
    cerr << "Error: profilee couldn't start its program!\n";
    perror(nullptr);
    exit(-1);
  } else { /* error */
    cerr << "Error: couldn't fork audited program.\n";
    return -1;
  }

}

