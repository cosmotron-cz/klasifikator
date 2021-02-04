var url = window.location.hostname;
var create = 'http://' + url + ':5001/new_planned_classification';
var read = 'http://' + url + ':5001/get_planned_classifications';
var update = 'http://' + url + ':5001/update_planned_classification';
var destroy = 'http://' + url + ':5001/delete_planed_classification';
Ext.define('ClassificationApp.store.PlannedClassifications', {
    extend: 'Ext.data.Store',

    alias: 'store.plannedclassifications',

    model: 'ClassificationApp.model.PlannedClassification',
    // data: [
    //     {id: 'asdf', data: 'data1', model: 'modelbla', note: 'poznamka', date: '2021-01-25 12:12', email: 'email@email.com', status: 'Planned', error: '', export: ''},
    // ],
	proxy: {
        type: 'ajax',
		api: {
			create  : create,
			read    : read,
			update  : update,
			destroy : destroy
        },
        actionMethods: {
            create: 'POST',
            read: 'GET',
            update: 'POST',
            destroy: 'POST'
        },
        writer: {
            type: 'json',
            writeAllFields: true,
        },
        reader: {
            type: 'json',
            rootProperty: 'data'
        }
    },
});