var url = window.location.hostname;
url = 'http://' + url + ':5001/get_models';
Ext.define('ClassificationApp.store.Models', {
    extend: 'Ext.data.Store',

    alias: 'store.models',

    fields:[ 'model' ],
	proxy: {
        type: 'ajax',
        url: url,
        reader: {
            type: 'json',
            rootProperty: 'data'
        }
    },
});