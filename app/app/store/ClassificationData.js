var url = window.location.hostname;
url = 'http://' + url + ':5001/get_classification_data';
Ext.define('ClassificationApp.store.ClassificationData', {
    extend: 'Ext.data.Store',

    alias: 'store.classificationdata',

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