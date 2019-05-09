var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

async function getModel(){
  const model = await tf.loadModel('https://gymnos/tfjs_files/model.json');
}

module.exports = router;
