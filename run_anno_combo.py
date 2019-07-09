# Annotator code for use with ris_widget, received from Will on 7/9/19

from ris_widget import ris_widget; rw = ris_widget.RisWidget()
from elegant.gui import pose_annotation
from elegant.gui import experiment_annotator
from elegant.gui import stage_field
from elegant import load_data

from elegant import clean_timepoint_data as ctd
import os

def filter_timepoint_wrap(exp_dir,filter_excluded = True,annotation_dir='annotations',channels=['bf']):

    positions = load_data.read_annotations(exp_dir,annotation_dir=annotation_dir)

    if filter_excluded: positions = load_data.filter_annotations(positions, load_data.filter_excluded)

    def timepoint_filter(position_name,timepoint_name):

        return os.path.exists(exp_dir+os.path.sep+position_name+os.path.sep+timepoint_name+' comp.png')

    def good_pos_filter(position_name,timepoint_name):

        if position_name in positions: return timepoint_filter(position_name,timepoint_name)

        else: return position_name in positions

    return load_data.scan_experiment_dir(exp_dir,channels=channels, timepoint_filter=good_pos_filter)


def replace_annotation(experiment_root, annotation_type, old_annotation_values, new_annotation_value, annotation_dir='annotations'):

    reinit = input('press y and enter to delete annotations; press enter to exit. ')

    if reinit.lower() == 'y':
        if not isinstance(old_annotation_values, collections.Iterable):
            old_annotation_values = list(old_annotation_values)
        if isinstance(old_annotation_values, str):
            old_annotation_values = [old_annotation_values]

        experiment_annotations = load_data.read_annotations(experiment_root, annotation_dir=annotation_dir)
        for position, position_annotations in experiment_annotations.items():
            for timepoint, timepoint_annotations in position_annotations[1].items():
                if annotation_type in timepoint_annotations and timepoint_annotations[annotation_type] in old_annotation_values:
                    timepoint_annotations[annotation_type] = new_annotation_value
        load_data.write_annotations(experiment_root, experiment_annotations, annotation_dir=annotation_dir)

    return

def filter_good_positions_wrap(experiment_root,channels='bf',error_on_missing=False):

    positions = load_data.read_annotations(experiment_root)

    good_positions = load_data.filter_annotations(positions, load_data.filter_good_incomplete)
    
    def timepoint_filter(position_name, timepoint_name):
        return position_name in good_positions

    return load_data.scan_experiment_dir(experiment_root, timepoint_filter=timepoint_filter,channels=channels,error_on_missing=error_on_missing)


def annotate_poses(exp_root,channels='bf'):

    ws = pose_annotation.PoseAnnotation(rw, name='pose', width_estimator=None, objective=10, optocoupler=0.7)

    sf = stage_field.StageField(stages=['egg', 'larva', 'adult', 'dead'])

    positions = load_data.scan_experiment_dir(exp_root,channels=channels,error_on_missing=False)

    ea = experiment_annotator.ExperimentAnnotator(rw, exp_root, positions,[sf,ws])

def annotate_stages(exp_root,channels='bf'):

    sf = stage_field.StageField(stages=['egg', 'larva', 'adult', 'dead'])

    positions = filter_good_positions_wrap(exp_root,channels=channels,error_on_missing=False)

    ea = experiment_annotator.ExperimentAnnotator(rw, exp_root, positions,[sf])#,start_position='021')



if __name__ == "__main__":


    #exp_root = '/Volumes/janewarray/Pittman_Will/20190329_1ul_ctrl'

    exp_root = '/Volumes/purplearray/Pittman_Will/20190521_cyclo_live/'

    annotate_poses(exp_root,channels='comp')

    #annotate_stages(exp_root,channels='bf')



    # ws = pose_annotation.PoseAnnotation(rw, name='pose', width_estimator=None, objective=10, optocoupler=0.7)

    # sf = stage_field.StageField(stages=['egg', 'larva', 'adult', 'dead'])

    # #  replace_annotation(exp_root,'stage',['dead','adult','larva'],'egg',annotation_dir='/Volumes/9karray/Pittman_Will/20181218_zpl12_5x/annotations_test')

    # #sf = stage_field.StageField(stages=['egg','l1','l2','l3','l4','young adult','adult', 'dead'],shortcuts=['e','z','x','c','v','b','a','d'])

    # #positions = load_data.scan_experiment_dir(exp_root,channels=channels,error_on_missing=False)

    # #positions = filter_timepoint_wrap(exp_root,channels=channels)

    # positions = filter_good_positions_wrap(exp_root,channels=channels,error_on_missing=False)

    # ea = experiment_annotator.ExperimentAnnotator(rw, exp_root, positions,[sf,ws])#,start_position='021')

    

    #ea.load_position_index(15)

