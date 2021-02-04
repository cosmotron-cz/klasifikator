var url = window.location.hostname;
url = 'http://' + url + ':5001/get_training_data';
Ext.define('ClassificationApp.store.TrainingData', {
    extend: 'Ext.data.Store',

    alias: 'store.trainingdata',

    fields:[ 'data' ],
	proxy: {
        type: 'ajax',
        url: url,
        reader: {
            type: 'json',
            rootProperty: 'data'
        }
	},
});