import sys
import math
import numpy as np
import pandas as pd
from stormpy import Rational
from tqdm import tqdm

import premise

sys.path.append("../../src")
from DataGeneration import DataGenerator
from Oracles import Oracle


class PremiseOracle(Oracle):

    def __init__(self, oracle_interface, trace_mapper, data_generator):
        self.oracle_interface = oracle_interface
        self.trace_mapper = trace_mapper
        self.data_generator = data_generator

    def membership(self, trace):
        trace = self.trace_mapper.trace_from_high_level_trace(trace)
        risk = self.oracle_interface.membership(trace, intermediate_results=True)
        # print(float(risk[-1]))
        return [1] if float(risk[-1]) > 0.5 else [0]

    def chernoffAnalyze(self, artifact, confidence, precision):

        aggregator = 0
        counterexamples = pd.DataFrame(
            columns=self.data_generator.training_data.columns
        )
        # computer number of sample
        num_traces = math.ceil(np.log(2 * confidence) / (2 * (precision**2)))
        print(f"generating {num_traces} test traces")
        traces = self.data_generator.generate_input_traces(
            num_traces, self.data_generator.trace_length
        )
        # print(traces)
        flattened_traces = self.data_generator.flatten_training_data(traces, False)

        for index, flattened_trace in tqdm(flattened_traces.iterrows()):
            artifact_verdict = artifact.predict([flattened_trace.values])
            trace = traces.loc[index]
            oracle_verdict = self.membership(trace)
            # print(int(artifact_verdict[0]),int(oracle_verdict[0]))
            if int(artifact_verdict[0]) == int(oracle_verdict[0]):
                aggregator += 1
            else:
                counterexample = np.append(trace.values, oracle_verdict)
                counterexamples.loc[len(counterexamples.index)] = counterexample

        aggregator = aggregator / (index + 1)
        print(f"Result: {aggregator}")

        return aggregator, counterexamples

    def analyze(self, monitor):
        aggregator, counterexample = self.chernoffAnalyze(monitor, 0.99, 0.01)
        terminate = False
        if aggregator > 0.98:
            terminate = True

        return aggregator, counterexample, terminate


class PremiseDataGenerator(DataGenerator):

    def __init__(self, training_data, premise_generator, trace_length, tracemapper):
        super().__init__(training_data)
        self.generator = premise_generator
        self.trace_length = trace_length
        self.trace_mapper = tracemapper

        # create training data signature
        self.sig = list(training_data.columns)
        self.input_sig = self.sig[:-1]
        self.label_sig = self.sig[-1:]

    def flatten_training_data(self, training_data, label=False):

        # create new column names from training_data and trace mapper
        feature_names = self.trace_mapper.get_observation_names()
        if label:
            training_data_names = list(training_data.columns)[:-1]
        else:
            training_data_names = list(training_data.columns)

        new_columns_names = []
        for i in training_data_names:
            for j in feature_names:
                new_columns_names.append(i + "_" + j)

        if label:  # TODO fix append label
            new_columns_names.append(list(training_data.columns)[-1])

        flattened_data = pd.DataFrame(columns=new_columns_names)

        for index, row in tqdm(training_data.iterrows()):
            new_row = []
            for i in list(training_data.columns):
                feature_item = row.loc[i]
                try:
                    for j in feature_item:
                        new_row.append(j)
                except:
                    new_row.append(feature_item)

            flattened_data.loc[len(flattened_data.index)] = new_row

        return flattened_data

    def generate_input_traces(self, num_traces, trace_length):

        traces = pd.DataFrame(columns=self.input_sig)

        for i in tqdm(range(num_traces)):
            trace = self.generator.generate_random_trace(trace_length)
            trace = self.trace_mapper.trace_to_high_level(trace)
            trace = pd.DataFrame([trace], columns=self.input_sig)
            traces = pd.concat([traces, trace], ignore_index=True)

        return traces

    def update_training_data(self, new_data):
        self.training_data = pd.concat(
            [self.training_data, new_data], ignore_index=True
        )

    def process_input_trace(self, input_traces):
        pass

    def process_counterexamples(self, counterexamples):
        self.update_training_data(counterexamples)

    def get_trace_length(self):
        return self.trace_length

    def get_training_data(self):
        flattened_data = self.flatten_training_data(self.training_data, True)
        return flattened_data
