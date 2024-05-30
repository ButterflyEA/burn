use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};
use super::confusion_matrix::ConfusionMatrix;

/// The precision metric.
#[derive(Default)]
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [precision metric](PrecisionMetric) input type.
#[derive(new)]
pub struct PrecisionInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> PrecisionMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the pad token.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }

    fn calc_precision(&self, cm: ConfusionMatrix<B>) -> f64 {
        if cm.matrix.dims() == [2,2] {
            let preicision = cm.matrix
                .clone()
                .slice([0..1, 0..1])
                .float() 
                / (cm.matrix
                    .clone()
                    .slice([0..1,0..1])
                    .float() 
                    + cm.matrix
                    .clone()
                    .slice([0..1,1..2])
                    .float());

            f64::from_elem(preicision.to_data().value[0])

        }
        else {
            let v = cm.split_cm();
            let l = v.len();
            let precision = v.into_iter()
                .map(|x| self.calc_precision(x)).sum::<f64>() / l as f64;
            
            precision
        }
    }
}

impl<B: Backend> Metric for PrecisionMetric<B> {
    const NAME: &'static str = "Precision";

    type Input = PrecisionInput<B>;

    fn update(&mut self, input: &PrecisionInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.clone().to_device(&B::Device::default());
        let outputs = input
            .outputs
            .clone()
            .argmax(1)
            .to_device(&B::Device::default())
            .reshape([batch_size]);

        let mut cm = ConfusionMatrix::<B>::new();
        _ = cm.from_outputs(&outputs, &targets, _n_classes);

        let precision = match self.pad_token {
            Some(pad_token) => {
                let mask = targets.clone().equal_elem(pad_token as i64);
                let matches = outputs.equal(targets).int().mask_fill(mask.clone(), 0);
                let num_pad = mask.int().sum().into_scalar().elem::<f64>();

                matches.sum().into_scalar().elem::<f64>() / (batch_size as f64 - num_pad)
            }
            None => {
                self.calc_precision(cm)
            }
        };
        println!("{}", precision);

        self.state.update(
            precision,
            batch_size,
            FormatOptions::new(Self::NAME).unit("").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for PrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_precision_without_padding() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new();
        let input = PrecisionInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8], // 2
                    [1.0, 2.0, 0.5], // 1
                    [0.4, 0.1, 0.2], // 0
                    [0.6, 0.7, 0.2], // 1
                ],
                &device,
            ),
            Tensor::from_data([2, 2, 1, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(0.5, metric.value());
    }

    #[test]
    fn test_precision_without_padding2() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new();
        let input = PrecisionInput::new(
            Tensor::from_data(
                [
                    [2., 3., 6.], //2
                    [4., 5., 3.], //1
                    [2., 7., 12.],//2
                    [22., 3., 6.], //0
                    [4., 5., 3.], //1
                    [22., 7., 12.] //0   
                ],
                &device,
            ),
            Tensor::from_data([1, 1, 2, 0, 1, 2], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!( 0.6666666666666666, metric.value());
    } 

    
    #[test]
    fn test_binary() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new();
        let input = PrecisionInput::new(
            Tensor::from_data(
                [
                    [2., 3.], //1
                    [4., 5.], //1
                    [2., 7.],//1
                    [22., 3.], //0
                    [4., 5.], //1
                    [22., 7.] //0   
                ],
                &device,
            ),
            Tensor::from_data([1, 1, 1, 0, 1, 0], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!( 1.0, metric.value());
    }     

    // #[test]
    // fn test_precision_with_padding() {
    //     let device = Default::default();
    //     let mut metric = PrecisionMetric::<TestBackend>::new().with_pad_token(3);
    //     let input = PrecisionInput::new(
    //         Tensor::from_data(
    //             [
    //                 [0.0, 0.2, 0.8, 0.0], // 2
    //                 [1.0, 2.0, 0.5, 0.0], // 1
    //                 [0.4, 0.1, 0.2, 0.0], // 0
    //                 [0.6, 0.7, 0.2, 0.0], // 1
    //                 [0.0, 0.1, 0.2, 5.0], // Predicted padding should not count
    //                 [0.0, 0.1, 0.2, 0.0], // Error on padding should not count
    //                 [0.6, 0.0, 0.2, 0.0], // Error on padding should not count
    //             ],
    //             &device,
    //         ),
    //         Tensor::from_data([2, 2, 1, 1, 3, 3, 3], &device),
    //     );

    //     let _entry = metric.update(&input, &MetricMetadata::fake());
    //     assert_eq!(50.0, metric.value());
    // }
}
