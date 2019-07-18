
use strict;
use warnings;

# Standard libraries:
use Getopt::Long;
use File::Basename;
use File::Copy;
use Data::Dumper;

# Runjobs libraries:
#use lib ("/d4m/ears/releases/Cube2/R2011_11_21/install-optimize$ENV{ARCH_SUFFIX}/perl_lib");
use lib ("/d4m/ears/releases/Cube2/R2016_02_12/" .
    "install-optimize$ENV{ARCH_SUFFIX}/perl_lib");
use runjobs4;
use File::PathConvert;
use Parameters;
use PerlUtils;
use RunJobsUtils;

use Batch;

# Package declaration:
package main;

our ($exp_root, $exp) = startjobs();
my $expts = "$exp_root/expts";

max_jobs(100,);

my $exptName = "argument_search_tst";

######## These are the things user might want to change ########
#my $SINGULARITY_GPU_QUEUE = "allGPUs-v100";
#$Batch::SINGULARITY_GPU_QUEUE = "allGPUs-sl69-non-k10s";
#$Batch::SINGULARITY_GPU_QUEUE = "allCPUs-sl69";
$Batch::SINGULARITY_GPU_QUEUE = "allGPUs-v100";

$Batch::SINGULARITY_WRAPPER = "$exp_root/scripts/run-in-singularity-container.sh";


# my @batch_sizes = (20,50,100);
# my @num_epochs = (5,35,50);
# my @positive_weights = (3,5,7);

# my @cnn_filter_lengths = ("3","3,4,5","2,3,4,5");
# my @embedding_vector_lengths = (20,50);
# my @neighbor_distances = (3);
# my @number_of_feature_maps = (150,200);
# my @num_fine_tune_epochs = (0);



my $exptDir = "$expts/$exptName";

if(!-d $exptDir) {
	my $cmd = "mkdir -p $exptDir";
	`$cmd`;
}

my @bool_to_str = ("false", "true");


my @run_list = generate_embedded_runs({
	exptDir => $exptDir
});


foreach my $entry (@run_list){
	
	my $training_jobid = Batch::execute_train_dev_test({
		run_dir                          => $entry->{run_dir},
		dependant_job_ids                => [], 
		job_name                         => $exptName . "/gpu/" . $entry->{name} . "/train", 
		mode                             => "train",
		extraction_type                  => "arg",
		train_mode                       => "train_argument",
		test_mode                        => "test_argument",
		model_type                       => $entry->{model_type},
		model_config_params              => $entry->{model_config_params}
												});
	
	my $test_jobid = Batch::execute_train_dev_test({
		run_dir                          => $entry->{run_dir},
		dependant_job_ids                => [$training_jobid], 
		job_name                         => $exptName . "/gpu/" . $entry->{name} . "/test", 
		mode                             => "test",
		extraction_type                  => "arg",
		train_mode                       => "train_argument",
		test_mode                        => "test_argument",
		model_type                       => $entry->{model_type},
		model_config_params              => $entry->{model_config_params}
											});
}


endjobs();

sub generate_embedded_runs {
	my ($args) = @_;	

	chomp(%$args);

	my @run_list = ();
	
	foreach my $use_position_feat (0, 1){

	foreach my $use_common_entity_name (0, 1){

    foreach my $use_dep_embedding (0, 1){

	foreach my $end_hidden_layer_nodes (64, 128){

    foreach my $run_num (0,1,2){
	my $name = "arg_embedded_" . 
		"pf" . $use_position_feat . "_" .
		"ce" . $use_common_entity_name . "_" .
		"dep". $use_dep_embedding . "_" .
		"h". $end_hidden_layer_nodes .  "_" .
	    "run". $run_num;
	my $entry = {
		name => $name,
		run_dir => $args->{exptDir} . "/". $name . "/",
		model_type => "embedded",
		model_config_params => {
			early_stopping => "true",
			batch_size => 50,
			num_epochs => 50,
			entity_embedding_vector_length => 50,
			neighbor_distance => 1,
			position_embedding_vector_length => 50,
			positive_weight => 1,
			end_hidden_layer_depth => 2,
			end_hidden_layer_nodes => $end_hidden_layer_nodes,
			use_end_hidden_layer => "true",
			use_position_feat => $bool_to_str[$use_position_feat],
			use_common_entity_name => $bool_to_str[$use_common_entity_name],
			use_dep_embedding => $bool_to_str[$use_dep_embedding]
		}
	};
	push @run_list, $entry;
    }
    }
    }
	}
	}
	return @run_list;
}

sub generate_cnn_runs {
	my ($args) = @_;	

	chomp(%$args);
	my @batch_sizes = (50);
	my @positive_weights = (3);

	my @cnn_filter_lengths = ("3,4,5");
	my @embedding_vector_lengths = (50);
	my @neighbor_distances = (3);
	my @number_of_feature_maps = (150);
	my @num_fine_tune_epochs = (0);

	my @run_list = ();

	foreach my $embedding_vector_length(@embedding_vector_lengths){

	foreach my $feature_maps(@number_of_feature_maps){
	
	foreach my $cnn_filter_length(@cnn_filter_lengths){

	foreach my $neighbor_distance(@neighbor_distances){
	
	foreach my $batch_size(@batch_sizes){
			
	foreach my $use_event_embedding(0, 1){

	foreach my $positive_weight(@positive_weights){
		my $name = "arg_cnn_" . 
			"w". $positive_weight . "_" .
			"cfl" . $cnn_filter_length =~ tr/\,/\_/r . "_" .
			"evl" . $embedding_vector_length . "_" .
			"nd" . $neighbor_distance . "_" .
			"f" . $feature_maps . "_" .
			"b" . $batch_size . "_" .
			"eemb". $use_event_embedding;

		my $entry = {
			name => $name,
			run_dir => $args->{exptDir} . "/". $name . "/",
			model_type => "cnn",
			model_config_params => {
				early_stopping => "true",
				batch_size => $batch_size,
				num_epochs => 50,
				entity_embedding_vector_length => $embedding_vector_length,
				neighbor_distance => $neighbor_distance,
				position_embedding_vector_length => $embedding_vector_length,
				positive_weight => $positive_weight,
				cnn_filter_length => $cnn_filter_length,
				use_event_embedding => $use_event_embedding,
				number_of_feature_maps => $feature_maps
			}
		};
		push @run_list, $entry;
	}
	}
	}
	}
	}
	}
	}
	return @run_list;
}
