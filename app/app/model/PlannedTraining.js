Ext.define('ClassificationApp.model.PlannedTraining', {
    extend: 'Ext.data.Model',

    fields: [
        { name: 'id', type: 'string'},
        { name: 'data', type: 'string'},
        { name: 'note', type: 'string'},
        { name: 'date', type: 'date', dateFormat: 'Y-m-d H:i'},
        { name: 'email', type: 'string'},
        { name: 'status', type: 'string'},
        { name: 'error', type: 'string'},
        { name: 'model', type: 'string'},
    ],
    idProperty: 'id'
});