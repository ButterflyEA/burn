use burn_core::tensor::{Tensor, Int, ElementConversion};
use burn_core::tensor::backend::Backend;


#[derive(Debug, Clone)]
pub struct ConfusionMatrix <B: Backend> {
    pub matrix: Tensor<B, 2, Int>,
}

impl <B: Backend> ConfusionMatrix <B> {
    pub fn new() -> Self {
        let matrix: Tensor<B, 2, Int> = Tensor::zeros([1,1], &B::Device::default());
        Self {matrix}
    }

    pub fn from_outputs(&mut self, outputs: &Tensor<B, 1, Int>, targets: &Tensor<B, 1, Int>, _n_classes: usize) -> ConfusionMatrix<B> {
         let [batch_size] = outputs.dims();
         let ee = outputs
                .clone()
                .equal(targets.clone())
                .argwhere()
                .to_data();

        let mut matrix = Tensor::<B, 2, Int>::zeros([_n_classes,_n_classes], &B::Device::default());

        let mut upd_mat = matrix.clone();

        for j in 0.. batch_size {
            let cls_pred = i64::from_elem(outputs.clone().to_data().value[j]) as usize;
            let cls_tgt = i64::from_elem(targets.clone().to_data().value[j]) as usize;

            if cls_pred == cls_tgt {
                let values = matrix
                    .clone()
                    .slice([cls_pred..cls_pred+1, cls_tgt..cls_tgt+1]) + 1;

                upd_mat = matrix
                    .slice_assign( [cls_pred..cls_pred+1, cls_tgt..cls_tgt+1], values); 
            }
            else {
                let values = matrix
                    .clone()
                    .slice([cls_pred..cls_pred+1, cls_tgt..cls_tgt+1]) + 1;

                upd_mat = matrix
                    .slice_assign( [cls_pred..cls_pred+1, cls_tgt..cls_tgt+1], values);                 
            }

            matrix = upd_mat.clone();        
        }
        self.matrix = matrix;
        //println!("{}", self.matrix);
        self.clone()
    }

    pub fn set(&mut self, t: Tensor::<B,2,Int>) {
        self.matrix = t;
    }

        /// Split confusion matrix in N one-vs-all binary confusion matrices
    pub fn split_cm(&self) -> Vec<ConfusionMatrix<B>> {
        let device = Default::default();
        let [dim1, _dim2] = self.matrix.dims();
        let sum = self.matrix.clone().sum();

        (0..dim1)
            .map(|i| {
                let tp = i32::from_elem(self.matrix
                    .clone()
                    .slice([i..i+1, i..i+1])
                    .to_data()
                    .value[0]);

                let fp = i32::from_elem(self.matrix
                    .clone()
                    .slice([i..i+1, 0..dim1]).sum().to_data().value[0])  - tp;

               let _fn = i32::from_elem(self.matrix
                .clone()
                .slice([0..dim1, i..i+1]).sum().to_data().value[0]) - tp;

                let tn = i32::from_elem(sum.to_data().value[0]) - tp - fp - _fn;

                let matrix = 
                    Tensor::<B, 2, Int>::from_ints([[tp, fp],[_fn, tn]], &device);

                println!("{}", matrix);
                
                let mut cm = ConfusionMatrix::<B>::new();
                cm.set(matrix.clone());

                cm
            })
            .collect()
        }
}