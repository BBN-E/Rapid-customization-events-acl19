use strict;
use warnings;

use lib ("/d4m/ears/releases/Cube2/R2017_07_21_1/install-optimize$ENV{ARCH_SUFFIX}/perl_lib");
use runjobs4;

package Batch;
require Exporter;
our @ISA = qw(Exporter);

our $SINGULARITY_GPU_QUEUE;

our $SINGULARITY_WRAPPER;

my $cmd_script = "/home/jfaschin/nfshome02/nlplingo_causeeffect/nlplingo/event/train_test.py";

my $python_path = "/nfs/raid87/u14/CauseEx/NN-events-requirements/SVN_PROJECT_ROOT_PY3:/home/jfaschin/nfshome02/nlplingo_causeeffect";


sub execute_train_dev_test {
	my ($args) = @_;	

	chomp(%$args);
	
	if(!-d $args->{run_dir}) {
		my $cmd = "mkdir -p " . $args->{run_dir};
		`$cmd`;
	}
	#print "GPU QUUE " . $SINGULARITY_GPU_QUEUE ."\n";
	my @run_sh_cmds = ();
	push @run_sh_cmds, "export KERAS_BACKEND=tensorflow\n";
	push @run_sh_cmds, "cd " . $args->{run_dir}. "\n";
	print $args->{mode} . "\n";
	if($args->{mode} eq "train"){
		push @run_sh_cmds, "cp \$1 train.params.json\n";
        #train_trigger_from_file
		push @run_sh_cmds, "PYTHONPATH=$python_path " . 
			"python3 $cmd_script --params \$1 --mode " . $args->{train_mode} . "\n";
	}elsif($args->{mode} eq "test"){
		#test_trigger
		push @run_sh_cmds, "cp \$1 test.params.json\n";
		push @run_sh_cmds, "PYTHONPATH=$python_path " . 
			"python3 $cmd_script --params \$1 --mode " . $args->{test_mode} . "\n";
		push @run_sh_cmds, "rm " . $args->{extraction_type} . ".hdf";
	}

	# Build the run.sh file
	my $bash_file = $args->{run_dir} . "/run_" . $args->{mode} . ".sh";
	open OUT, ">$bash_file";         
	
	for my $run_sh_cmd (@run_sh_cmds) {
		print OUT $run_sh_cmd;
	}
	close OUT;
	
	my $cmd = "chmod +x $bash_file";
	`$cmd`;
	
	# Schedule the run.sh file
	$cmd = "$bash_file";

	my %run_params = (
	        model_output_path                => $args->{run_dir} . "/" . $args->{extraction_type} . ".hdf",
		 	test_score_file_path             => $args->{run_dir} . "/test.score",
		 	train_score_file_path            => $args->{run_dir} . "/train.score",
			BATCH_QUEUE                      => $SINGULARITY_GPU_QUEUE, 
		 	SGE_VIRTUAL_FREE                 => "32G"
	);

	@run_params{keys %{$args->{model_config_params}}} = values %{$args->{model_config_params}};

	while ((my $k, my $v) = each %run_params) {
		print "$k => $v\n";
	}

	my $jobid = runjobs4::runjobs(
		$args->{dependant_job_ids}, 
		$args->{job_name}, 
		\%run_params, 
		["$SINGULARITY_WRAPPER $bash_file", 
		 $args->{extraction_type} . "." . $args->{mode} . "." . $args->{model_type} . ".template.params.json"]
		);
	return $jobid;
}

1;
